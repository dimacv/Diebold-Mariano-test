# Импорт нербходимых библиотек
import numpy as np
from scipy.stats import t
import collections
from re import compile as re_compile
import statsmodels.api as sm

# Класс, реализующий тест Дибольда-Мариано для сравнения точности двух наборов прогнозов(1995),
# с модификацией, предложенной Харви и др.(1997).
class Diebold_Mariano_Test:
    '''
    Параметры:
        crit :  строка, определяющая критерий,
            может принимать следующие значения -
                MSE : среднеквадратическая ошибка
                MAE или MAD : средняя абсолютная ошибка или среднее абсолютное отклонение
                MAPE : средняя абсолютная процентная ошибка
                MASE : средняя абсолютная масштабированная ошибка
                MRAE : Средняя относительная абсолютная ошибка
                SMAPE : Симметричная средняя абсолютная процентная ошибка
                poly : использование степенной функции для взвешивания ошибок
                ALL : рсчет по всем предыдущим критериям
        power : степень в функции для взвешивания ошибок(имеет смысл, только когда crit="poly")
        h : количество шагов вперед
        seasonal_period : количество периодов в полном сезонном цикле(например, 4 - для квартальных данных,
                         или 7 - для ежедневных данных с недельным циклом), используется только при расчете "MASE"
    Условия :
        1) h  должно быть целым числом и должно быть больше 0 и меньше длины actual_lst.
        2) crit должен принимать только значения, указанные выше.
        4) Каждое значение actual_lst, pred1_lst и pred2_lst должно быть числовым. Недостающие значения не принимаются.
        5) power - должна быть числовой величиной.
    '''

    # все параметры для инициализации публичных атрибутов
    # задаем в методе __init__
    def __init__(self, crit="MSE", power = 2, h = 1, seasonal_period=1 ):
        # строка, определяющая критерий
        self.crit = crit
        # количество шагов вперед
        self.h = h
        # степень в функции для взвешивания ошибок(имеет смысл, только когда crit="poly")
        self.power = power
        # количество периодов в полном сезонном цикле, используется только при расчете "MASE"
        self.seasonal_period = seasonal_period
        # предварительная установка признака того, что входные данные являются массивами numpy
        self.numpy_flag = True

    # метод возвращающий True, если значение соответствует шаблону регулярного выражения
    def compiled_regex(self, s, comp):
        # проверка соответствия строки s шаблону регулярного выражения comp
        if comp.match(s) is None:
            # если да, то возвращает признак, указывающий на то, содержит ли строка только цифры.
            return s.isdigit()
        return True

    # метод проверки наличия ошибок во входных данных
    def error_check(self, actual_lst, pred1_lst, pred2_lst):
        # установка признака ошибки в 0
        rt = 0
        # присвоение переменной сообщения пустого значения
        msg = ""
        # проверка - являяется ли значение self.h целочисленным
        if (not isinstance(self.h, int)):
            # установка признака ошибки в -1
            rt = -1
            # сообщение о том, что h не является целочисленным
            msg = "The type of the number of steps ahead (h) is not an integer."
            # возвращение признака ошибки и сообшения об ошибке
            return (rt, msg)
        # Проверка диапазона h
        if (self.h < 1):
            # если h отрицательное, то установка признака ошибки и соответствующего сообщения
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            # возвращение признака ошибки и сообшения об ошибке
            return (rt, msg)
        # вычисление длины входных данных
        len_act = len(actual_lst)
        len_p1 = len(pred1_lst)
        len_p2 = len(pred2_lst)
        # Проверка того, равны ли длины фактических и прогнозируемых значений.
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            # если нет, то установка признака ошибки и соответствующего сообщения
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            # возвращение признака ошибки и сообшения об ошибке
            return (rt, msg)
        # Проверка того, не превышает ли h длину входных данных
        if (self.h >= len_act):
            # если нет, то установка признака ошибки и соответствующего сообщения
            rt = -1
            msg = "The number of steps ahead is too large."
            # возвращение признака ошибки и сообшения об ошибке
            return (rt, msg)
        # Проверка на правильность значения критерия
        if (self.crit != "MSE" and self.crit != "MAPE" and self.crit != "MAD" and self.crit != "MAE" and self.crit != "poly"
                and self.crit != "MASE" and self.crit != "MRAE" and self.crit != "SMAPE" and self.crit != "ALL" ):
            # если нет, то установка признака ошибки и соответствующего сообщения
            rt = -1
            msg = "The criterion is not supported."
            # возвращение признака ошибки и сообшения об ошибке
            return (rt, msg)

        # передача функции re_compile соответствующего регулярного выражения для проверки -
        # является ли каждое значение входных списков числовым значением
        comp = re_compile("[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

        # если входные данные не являются массивами numpy, т.е. являются списками,
        if not (isinstance(actual_lst, np.ndarray) and isinstance(pred1_lst, np.ndarray) \
                and isinstance(pred2_lst, np.ndarray) ):
                    # если данные не numpy выставляется флаг
                    self.numpy_flag = False
                    # то в цикле проверяем каждое из значений списков
                    for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
                        # на предмет - являются ли они числовыми значениями
                        is_actual_ok = self.compiled_regex(str(abs(actual)), comp)
                        is_pred1_ok = self.compiled_regex(str(abs(pred1)), comp)
                        is_pred2_ok = self.compiled_regex(str(abs(pred2)), comp)
                        # если не являются, то установка признака ошибки и соответствующего сообщения
                        if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                            msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                            rt = -1
                            # возвращение признака ошибки и сообшения об ошибке
                            return (rt, msg)
        # Если ошибок нет, то возвращение признака ошибки равного 0 и пустого сообшения об ошибке
        return (rt, msg)

    # метод вычисляющий значения теста Дибольда-Мариано для заданных наборов данных
    def DBTest(self, actual_lst, pred1_lst, pred2_lst) -> object:
        '''
        actual_lst: список реальных значений зависимой переменной
        pred1_lst : первый список прогнозируемых значений
        pred2_lst : второй список прогнозируемых значений
        Условия:
            1) длины actual_lst, pred1_lst и pred2_lst должны быть равны
            2) actual_lst, pred1_lst и pred2_lst - должны быть либо числовыми массивами numpy,
               либо списками каждое значение которых должно быть числовым. Недостающие значения не принимаются.
        Возвращаемое значение:
            именованный кортеж(named-tuple) из 2 элементов :
            1) DM      :  тестовая статистика DM-теста.
            2) p_value : p-значение DM-теста.
        '''
        # создание пустых списков
        e1_lst = []
        e2_lst = []
        d_lst = []
        # проверка на ошибки во входных данных
        error_code = self.error_check(actual_lst, pred1_lst, pred2_lst)
        # выдача соответствующей ошибки, если есть признак ошибки
        if (error_code[0] == -1):
            raise SyntaxError(error_code[1])
            return

        # если входные данные не являются массивами numpy(т.е. являются списками)
        if not self.numpy_flag :
            # преобразовать каждое значение списков в вещественное число
            actual_lst = np.array(actual_lst).astype(np.float32).tolist()
            pred1_lst = np.array(pred1_lst).astype(np.float32).tolist()
            pred2_lst = np.array(pred2_lst).astype(np.float32).tolist()
        # длина списков (в виде действительного числа)
        T = float(len(actual_lst))

        # расчет значений теста Дибольда-Мариано в соответствии с различными критериями:

        # если критерий MSE
        if (self.crit == "MSE"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # вычисляем метрику качества MSE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append((actual - p1) ** 2)
                e2_lst.append((actual - p2) ** 2)
        # если критерий MAE или MAD
        elif (self.crit == "MAE" or self.crit == "MAD"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # вычисляем метрику качества MAE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append(abs(actual - p1))
                e2_lst.append(abs(actual - p2))
        # если критерий
        elif (self.crit == "MAPE"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # если значение actual равно нулю
                if actual == 0:
                    # присваить переменной идущей в знаменатель малое значение не равное нулю,
                    # во избежании получения ошибки деления на нуль
                    dv = 0.000001
                else:
                    # иначе в знаменатель идет значение actual
                    dv = actual
                # вычисляем метрику качества MAPE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append(100 * abs((actual - p1) / dv))
                e2_lst.append(100 * abs((actual - p2) / dv))
        # если критерий poly
        elif (self.crit == "poly"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # вычисляем метрику с использованием степенной функции для взвешивания ошибок
                # и добавляем в соответствующие списки метрик
                e1_lst.append(((actual - p1)) ** (self.power))
                e2_lst.append(((actual - p2)) ** (self.power))
        # если критерий MASE :
        # числителем тут будет абсолютная ошибка(MAE) прогноза данного момента времени,
        # а знаменателем - средняя абсолютная ошибка полученная на обучающей выборке
        # с помощью одношагово метода наивного сезонного прогноза
        elif (self.crit == "MASE"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # одношаговый метод наивного сезонного прогноза, это значение actual_lst[:-self.seasonal_period].
                # вычисление средней абсолютной ошибка полученная с помощью одношагово метода наивного сезонного
                # прогноза, идущего в знаменатель :
                dv = np.mean(np.abs(np.array(actual_lst[self.seasonal_period:]) -
                                         np.array(actual_lst[:-self.seasonal_period])))
                # вычисляем метрику качества MASE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append(abs((actual - p1) / dv))
                e2_lst.append(abs((actual - p2) / dv))
        # если критерий MRAE :
        elif (self.crit == "MRAE"):
            # в цикле перебираем значения элементов входных данных
            for cnt, (actual, p1, p2) in enumerate(zip(actual_lst, pred1_lst, pred2_lst)):
                # если это первая итерация
                if cnt == 0:
                    # запоминаем значение текущего элемента из actual_lst
                    actual_pre = actual
                    # и переходим к следующей итерации цикла
                    continue
                # формируем знаменатель как разницу текущего значения из actual_lst и прошлого
                dv = actual - actual_pre
                # если найденное значение равно нулю, заменяем его на малое значение не равное нулю,
                # во избежании ошибки деления на нуль
                if dv == 0: dv = 0.000001
                # вычисляем метрику качества MRAE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append(abs((actual - p1) / dv ))
                e2_lst.append(abs((actual - p2) / dv ))
                # запоминаем значение текущего элемента из actual_lst
                actual_pre = actual
        # если критерий SMAPE :
        elif (self.crit == "SMAPE"):
            # в цикле перебираем значения элементов входных данных
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                # формируем первый знаменатель
                dv1 = abs(actual + p1)
                # формируем второй знаменатель
                dv2 = abs(actual + p2)
                # если их значение равно нулю, заменяем его на малое но не нулевое значение
                if dv1 == 0: dv1 = 0.000001
                if dv2 == 0: dv2 = 0.000001
                # вычисляем метрику качества SMAPE для каждого из элементов и добавляем в соответствующие списки метрик
                e1_lst.append(200 * abs((actual - p1) / dv1))
                e2_lst.append(200 * abs((actual - p2) / dv2))
        # Если критерием является "ALL" - вычисляем значения тестов Дибольда-Мариано со всеми критериями "MSE","MAPE",
        # "MAE" и т.д.
        # В данном случае, возвращемое значение будет словарем с соответствующими ключами
        # и значениями  -  именованными кортежами возвращаемыми каждым отдельным методом
        if (self.crit == "ALL"):
            # создание пустого словаря
            rt = {}
            # создание списка с именами всех метрик
            cr = ["MSE","MAPE","MAE","MASE","MRAE","SMAPE","poly"]
            # прохождение в цикле по этому списку
            for i in cr:
                # рекурсивное создание экземпляра класса Diebold_Mariano_Test с параметром метрики в текущей итерации
                dmt = Diebold_Mariano_Test(crit=i)
                # расчет метрики теста Дибольда-Мариано с соответствующей метрикой
                # и занесение ее в результативный словарь
                rt[i] = dmt.DBTest(actual_lst, pred1_lst, pred2_lst)
                # удаление экземпляра класса Diebold_Mariano_Test
                del dmt
            # возращаем в качестве результата словарь
            return rt

        # если критерий был не ALL, вычисляем попарно разницы найденных метрик для каждого из элементов
        # для каждого момента времени
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

        # вычисляем среднее значение
        mean_d = np.array(d_lst).mean()

        # нахождение автоковариации, используя соответствующую функцию из statsmodels
        #d_cov = sm.tsa.acovf(d_lst, nlag=self.h - 1, fft=False, missing='drop')
        d_cov = sm.tsa.acovf(d_lst, nlag=self.h - 1, fft=True)
        # промежуточное значение при расчете тестовой статистике DM-теста
        V_d = (d_cov[0] + 2 * d_cov[1:].sum()) / len(d_lst)
        # и расчет тестовай статистики DM-теста
        DM_stat = V_d ** (-0.5) * mean_d
        # реализация модификациии теста предложенной Харви и др. в 1997г. :
        # расчет коэфициэнта
        harvey_adj = ((T + 1 - 2 * self.h + self.h * (self.h - 1) / T) / T) ** (0.5)
        # поправка статистики умножением на этот коэфициэнт
        DM_stat = harvey_adj * DM_stat
        # рсчет  p-значения
        p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

        # Создание экземпляра именованного кортежа для возврата
        dm_return = collections.namedtuple('dm_return',  ['DM', 'p_value'])
        rt = dm_return(DM=DM_stat, p_value=p_value)
        # возврат именованного кортежа с найденными значениями теста Дибольда-Мариано
        return rt

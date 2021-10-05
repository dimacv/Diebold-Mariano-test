# Diebold-Mariano-test

Класс Diebold_Mariano_Test, реализующий тест Дибольда-Мариано(1995) для статистического определения эквивалентности точности прогнозов для 2 наборов прогнозов -
https://www.sas.upenn.edu/~fdiebold/papers/paper68/pa.dm.pdf

с модификацией, предложенной Харви и др. al (1997) -
http://www.phdeconomics.sssup.it/documents/Lesson19.pdf (стр.19).

Сделан на основе переработанной этой функции - https://github.com/johntwk/Diebold-Mariano-Test/blob/master/dm_test.py

Сделана некоторое улучшение кода(к примеру функция автокореляции берется из statsmodels и т.д. и т.п.),
добавлены возможности использования в виде критериев дополнительных метрик( теперь они могут быть - MSE, MAE или MAD, MAPE, MASE, MRAE, SMAPE, poly, ALL ).

Возвращает либо именованный кортеж с найденными значениями теста Дибольда-Мариано( DM_stat, p_value ).

Либо, в случае вызова с  критерием "ALL", вычисляются значения тестов Дибольда-Мариано со всеми критериями - MSE, MAE или MAD, MAPE, MASE, MRAE, SMAPE, poly.
И в данном случае, возвращемое значение будет словарем с соответствующими ключами( "MSE", "MAE", и т.д.) и значениями  -  именованными кортежами возвращаемыми каждым отдельным методом.

---

References
Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. Journal of Business and Economic Statistics, 13, 253-263.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of forecasting, 13(2), 281-291.

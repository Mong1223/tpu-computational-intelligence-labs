# Методы вычислительного интеллекта (ТПУ, магистратура, 3 семестр)

| Папка | Содержание |
|---|---|
| `lab/lab1/laba` | Полносвязная нейросеть с нуля на NumPy (`nnClass.py`: прямое/обратное распространение) для классификации датасета Heart Disease, k-fold выборки (`gsamples.py`) |
| `lab/lab1/labaB` | Модифицированный вариант сети (`nnClassB.py`) |
| `lab/lab1/*.py` | Эксперименты с датасетами Diabetes / Heart, сравнение с SVM |
| `binary-classification` | Бинарная классификация планарных данных нейросетью с одним скрытым слоем |
| `practice/Task1` | Линейная регрессия и стохастический градиентный спуск (`PR1-VMI-Mongush.ipynb`) |
| `practice/Task2` | Переобучение и регуляризация; метрики качества классификации (precision/recall/ROC) |

## Запуск
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
cd lab/lab1/laba && python main.py
```

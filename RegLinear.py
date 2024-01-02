# Importando bibliotecas necessárias
from pydataset import data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Carregando o conjunto de dados do banco de Duncan
duncan = data('Duncan')

# Selecionando apenas variáveis quantitativas para a análise
quantitative_variables = ['income', 'education', 'prestige']

# Calculando e exibindo a matriz de correlação
correlation_matrix = duncan[quantitative_variables].corr()
print("Matriz de Correlação:")
print(correlation_matrix)

# Iterando sobre pares de variáveis para criar gráficos de dispersão com reta de regressão
for i in range(len(quantitative_variables)):
    for j in range(i+1, len(quantitative_variables)):
        x_variable = quantitative_variables[i]
        y_variable = quantitative_variables[j]

        # Criando o gráfico de dispersão
        sns.scatterplot(x=x_variable, y=y_variable, data=duncan)

        # Ajustando uma regressão linear
        model = LinearRegression()
        X = duncan[x_variable].values.reshape(-1, 1)
        y = duncan[y_variable].values
        model.fit(X, y)

        # Plotando a reta de regressão
        plt.plot(X, model.predict(X), color='red')

        # Calculando o coeficiente de determinação
        r2 = r2_score(y, model.predict(X))

        # Exibindo a expressão da reta de regressão estimada e o coeficiente de determinação
        equation = f"{y_variable} = {model.coef_[0]:.2f}*{x_variable} + {model.intercept_:.2f}"
        plt.title(f'Reta de Regressão: {equation}, R² = {r2:.2f}')

        # Exibindo o gráfico
        plt.show()

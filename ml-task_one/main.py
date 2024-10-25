import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Data/Nairobi Office Price Ex.csv')

# MSE function
def mean_squared_error(m, c, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].SIZE
        y = points.iloc[i].PRICE
        total_error += (y - (m * x + c)) ** 2
    return total_error / float(len(points))

# Gradient Descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].SIZE
        y = points.iloc[i].PRICE
        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    c = b_now - b_gradient * L
    return m, c

# Main function to initialize the linear regression
def main():
    m = np.random.rand()
    c = np.random.rand()
    L = 0.0001  # Learning rate
    epoch = 10  # Number of epochs

    for i in range(epoch):
        mse = mean_squared_error(m, c, data)
        print(f"Epoch: {i}, MSE: {mse}")

        m, c = gradient_descent(m, c, data, L)

    print(f"Final Slope (m): {m}, Final Intercept (c): {c}")

    # Plotting the line of best fit
    plt.scatter(data.SIZE, data.PRICE, color='black')
    x_values = data.SIZE
    y_values = [m * x + c for x in x_values]
    plt.plot( x_values, y_values, color='red')
    plt.xlabel("Office Size (sq. ft.)")
    plt.ylabel("Office Price")
    plt.title("A graph of Office Size in sq. ft. against Office Price")
    plt.show()

    # Office price be when the size is 100 sq. ft
    size_100_pred = m * 100 + c
    print(f"For office size 100 sq. ft, the price will be: {size_100_pred:.4f}")

# Run the main function
if __name__ == "__main__":
    main()

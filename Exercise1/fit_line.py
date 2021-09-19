BRUTEFORCE = False

# Linear solver
def mylinfit( x, y):

    #calculate a
    n = x.size
    sumX = np.sum(x)
    sumY = np.sum(y)

    sumXSquared = 0
    for i in range(0, n):
        sumXSquared += x[i] * x[i]

    sumXY = 0
    for i in range(0, n):
        sumXY += (x[i]*y[i])

    sulut = (sumX * sumXY) - (sumXSquared * sumY)
    jakaja = (sumX*sumX) - (n * sumXSquared)
    yhdistys = (sumX * sulut) / jakaja

    a = ((sumXY) - yhdistys) / sumXSquared

    #calculate b
    b = ((sumX*sumXY)-(sumXSquared*sumY)) / ((sumX*sumX)-(n*sumXSquared))

    return a, b

#bruteforces a,b values from -10 to 10 with 0.1 step
def bruteForceLinFit(x, y):
    aNumbers = np.linspace(-10, 10, 201)
    bNumbers = np.linspace(-10, 10, 201)
    mse = 100000
    tempmse = 0
    temp = 0
    besta = 0
    bestb = 0

    for a in aNumbers:
        for b in bNumbers:
            for i in range(0, y.size):
                temp = y[i] - (a * x[i] + b)
                temp = temp * temp
                tempmse += temp

            tempmse = tempmse / y.size

            if(tempmse < mse):
                mse = tempmse
                besta = a
                bestb = b

    return besta, bestb

#click event handler
def onClick(event):
    if event.button is MouseButton.LEFT:
        if event.inaxes:
            plotNewPoint(event.xdata, event.ydata)
            saveNewPoint(event.xdata, event.ydata)
    elif event.button is MouseButton.RIGHT:
        #configure line
        lineStart = -5
        lineEnd = 20
        xp = np.arange(lineStart, lineEnd, 0.1)

        # calculate and draw line math style (and bruteforce if True)
        a, b = mylinfit(xCoords, yCoords)
        print(f"Myfit : a={a} and b={b}")
        plt.plot(xp, a * xp + b, 'r-')

        if(BRUTEFORCE):
            aBrute, bBrute = bruteForceLinFit(xCoords, yCoords)
            print(f"BruteForceFit : a={aBrute} and b={bBrute}")
            plt.plot(xp, aBrute * xp + bBrute, 'y-')

    #update
    plt.draw()

def plotNewPoint(x,y):
    plt.plot(x, y, 'kx')

def saveNewPoint(x,y):
    global xCoords, yCoords
    xCoords = np.append(xCoords, [x])
    yCoords = np.append(yCoords, [y])

# Main
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np

#one random point
xCoords = np.random.uniform( -2, 5, 1)
yCoords = np.random.uniform( 0, 3, 1)

#axis limits
plt.xlim([-5,10])
plt.ylim([-5,10])

#plot one point
plt.plot( xCoords, yCoords, 'kx' )

#connect mouse clicked listener to plot
plt.connect('button_press_event', onClick)

plt.title("red line=MATH, yellow line=BRUTEFORCE(off by default)")

plt.figtext(0.5, 0.01, "left click = add points, right click = draw line", ha="center", fontsize=12)

plt.show()

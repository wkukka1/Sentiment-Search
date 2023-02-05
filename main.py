import PySimpleGUI as sg
from SentimentAnalysis import SentimentAnalysis, getAvg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas


def draw_figure(canvas, figure):
    fig = FigureCanvasTkAgg(figure, canvas)
    fig.draw()
    fig.get_tk_widget().pack(side="top", fill="both", expand=1)
    return fig


if __name__ == "__main__":
    query = ""
    sg.theme('BrownBlue')  # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Sentiment Search'), sg.InputText()],
              [sg.Button('Search'), sg.Button('Close Window')]]

    # Create the Window
    window = sg.Window('Test', layout).Finalize()

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (None, 'Close Window'):  # if user closes window or clicks cancel
            break
        if event == "Search":
            query = values[0]
            break

    window.close()
    layout = [
        [sg.Canvas(size=(1000, 1000), key="-CANVAS-")],
        [sg.Exit()]
    ]
    df = SentimentAnalysis(query)
    avg = getAvg(df)
    avg[0] = int(100 * avg[0])
    avg[1] = int(100 * avg[1])
    avg[2] = int(100 * avg[2])

    while sum(avg) != 100:
        avg[1] += 1
    avg = np.array(avg)
    mylabels = ["Positive", "Nuetral", "Negative"]
    plt.pie(avg, labels=mylabels)
    plt.show()

    # Creates a timeline of sentiment Analysis
    # dates = {}
    # for index, row in df.iterrows():
    #     dates.union(row["Date"])

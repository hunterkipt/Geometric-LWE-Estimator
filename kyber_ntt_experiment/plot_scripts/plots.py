from json import loads, dumps
import matplotlib.pyplot as plt
import numpy as np

# Requires counts file
def success_barplot(filename):
    # with open(filename) as f:
    #     outcomes = np.array(loads(f.read()))
    n = 64 - 40 + 1
    success_counts = np.empty((n,))
    for i in range(n):
        with open(f"counts_{40 + i}", "r") as fp:
            success_count[i] = loads(fp.read())["1.0"]


    x_guesses = np.arange(40, 64 + 1)
    y_succ = {
        'Fail': (100 - success_count, 'r'),
        'Success': (success_count, 'g'),
    }

    width = 0.9
    fig, ax = plt.subplots()
    bottom = np.zeros(len(x_guesses))

    for outcome, (count, c) in y_succ.items():
        p = ax.bar(x_guesses, count, width, label=outcome, bottom=bottom, color=c)
        bottom += count
        
        if outcome == "Fail":
            ax.bar_label(p, label_type='center')
        
    ax.invert_xaxis()
    ax.set_title('Attack Success Rate')

    ax.legend()

    plt.savefig("success.png")


# adapted from: https://stackoverflow.com/questions/16592222/how-to-create-grouped-boxplots
def accuracy_boxplot():
    prediction = []
    actual = []
    ticks = list(map(int, np.arange(64, 39, -1)))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
    for i in range(64, 51, -1):
        with open(f"results_{i}.json") as f:
            data = loads(f.read())
            bkz_ceil = data[0]["est"]["dim"]
            beta_array = np.array(data[-1]).T
            fails = 100 - beta_array.shape[1]
            prediction.append(beta_array[0])
            actual.append(beta_array[1])
            # actual.append(
            #     np.concatenate(
            #         [beta_array[1], np.array([bkz_ceil] * fails)]
            #     )
            # )
    bpl = ax_top.boxplot(prediction,
                         positions=np.array(range(len(prediction) - 1, -1, -1))*2.0-0.4, 
                         sym=' ', widths=0.6)
    bpr = ax_top.boxplot(actual, 
                         positions=np.array(range(len(actual) - 1, -1, -1))*2.0+0.4, 
                         sym=' ', widths=0.6)
    ax_top.set_xticks(range(len(ticks[:13]) * 2 - 2, -2, -2), ticks[:13])
    ax_top.set_xlim(-2, len(ticks[:13])*2)
    ax_top.set_ylim(0, 85)
    ax_top.invert_xaxis()
    prediction = []
    actual = []
    for i in range(51, 39, -1):
        with open(f"results_{i}.json") as f:
            data = loads(f.read())
            bkz_ceil = data[0]["est"]["dim"]
            beta_array = np.array(data[-1]).T
            fails = 100 - beta_array.shape[1]
            prediction.append(beta_array[0])
            actual.append(beta_array[1])
            # actual.append(
            #     np.concatenate(
            #         [beta_array[1], np.array([bkz_ceil] * fails)]
            #     )
            # )
    bpl = ax_bottom.boxplot(prediction, 
                            positions=np.array(range(len(prediction) - 1, -1, -1))*2.0-0.4, 
                            sym=' ', widths=0.6)
    bpr = ax_bottom.boxplot(actual, 
                            positions=np.array(range(len(actual) - 1, -1, -1))*2.0+0.4, 
                            sym=' ', widths=0.6)
    ax_bottom.set_xticks(range(len(ticks[13:]) * 2 - 2, -2, -2), ticks[13:])
    ax_bottom.set_xlim(-2, len(ticks[13:])*2)
    ax_bottom.set_ylim(0, 85)
    ax_bottom.invert_xaxis()
    fig.suptitle("Beta Estimation Accuracy")
    plt.savefig("average_bkz.png")

# averages
def accuracy_plot():
    x = np.arange(64, 39, -1)
    prediction = np.zeros((25,))
    actual = np.zeros((25,))
    actual_success = np.zeros((25,))
    for i in range(64, 39, -1):
      with open(f"results_{i}.json") as f:
        data = loads(f.read())
        bkz_ceil = data[0]["est"]["dim"]
        beta_array = np.array(data[-1]).T
        fails = 100 - beta_array.shape[1]
        prediction[64 - i] = np.average(beta_array[0])
        actual_success[64 - i] = np.average(beta_array[1])
        actual[64 - i] = (np.sum(beta_array[1]) + (fails * bkz_ceil)) / 100
    fig, ax = plt.subplots()
    ax.plot(x, prediction, 'o-', label="Prediction")
    ax.plot(x, actual_success, 'o-', label="Actual | Success")
    ax.plot(x, actual, 'o-', label="Actual")
    ax.invert_xaxis()
    ax.set_title("Beta Predictions Including Failures")
    ax.legend()
    plt.savefig("averages_total.png")

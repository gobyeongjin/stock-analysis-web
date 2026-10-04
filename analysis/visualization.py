import matplotlib.pyplot as plt


def plot_all(data, ticker):
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 12)
    )

    # 1. 주가 + 이동평균선
    axes[0].plot(
        data.index,
        data["Close"],
        label=ticker
    )

    axes[0].plot(
        data.index,
        data["MA20"],
        label="MA20"
    )

    axes[0].plot(
        data.index,
        data["MA60"],
        label="MA60"
    )

    axes[0].set_title(f"{ticker} Stock Price")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid()

    # 2. RSI
    axes[1].plot(
        data.index,
        data["RSI"],
        label="RSI"
    )

    axes[1].axhline(
        70,
        linestyle="--",
        label="Overbought (70)"
    )

    axes[1].axhline(
        30,
        linestyle="--",
        label="Oversold (30)"
    )

    axes[1].set_title(f"{ticker} RSI")
    axes[1].set_ylabel("RSI")
    axes[1].legend()
    axes[1].grid()

    # 3. 거래량
    axes[2].bar(
        data.index,
        data["Volume"],
        label="Volume"
    )

    axes[2].plot(
        data.index,
        data["Volume_MA20"],
        label="Volume MA20"
    )

    axes[2].set_title(f"{ticker} Trading Volume")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Volume")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()
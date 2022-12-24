import mplfinance as mpf


def PlotTrade(trade, trades_df, window_size, entry_candle, budget, sentiment):

    # try:
    #     trades_df = trades_df.drop('APISentiment', axis=1)
    # except:
    #     pass
    mover = 0
    if sentiment == True:
        mover = 1

    Dates = trades_df['Datetime']

    print("Trade: ", trade)

    selected_df = trades_df[trades_df['trade'] == trade]

    # Get EMAs df.iloc[:,4]
    dates = selected_df['Datetime']
    ema6 = selected_df.iloc[:, 4]  # selected_df['EMA6']
    ema6 = ema6[:-1]
    ema12 = selected_df.iloc[:, 5]  # selected_df['EMA12']
    ema12 = ema12[:-1]
    ema24 = selected_df.iloc[:, 6]  # selected_df['EMA24']
    ema24 = ema24[:-1]

    datepairs_ema6 = [(d1, d2) for d1, d2 in zip(dates, ema6)]
    datepairs_ema12 = [(d1, d2) for d1, d2 in zip(dates, ema12)]
    datepairs_ema24 = [(d1, d2) for d1, d2 in zip(dates, ema24)]

    # #Format Dataframe
    quotes = selected_df.iloc[:, :10+mover]
    quotes['Datetime'] = quotes['Datetime'].astype('datetime64')
    quotes = quotes.set_index('Datetime')
    # quotes.iloc[-1,10]
    # quotes.iloc[-1,0]
    # dicti = {'Price':[quotes.iloc[-1,0]],'Datetime':[quotes.iloc[-1,10]]}
    #df__ = pd.DataFrame(dicti)
    if sentiment == True:
        sentim = quotes.iloc[:, 9]

    quotes = quotes.iloc[:, :4]
    quotes.columns = ['open', 'high', 'low', 'close']

    # Define function to get entry
    EntryPriceRow = 0

    def GetEntryPriceColl(candle):
        if candle == 'Previous High':
            EntryPriceColl = 1
            EntryPriceRow = 1
        if candle == 'Current Open':
            EntryPriceColl = 0
            EntryPriceRow = 0
        if candle == 'Previous Close':
            EntryPriceColl = 3
            EntryPriceRow = 1
        return EntryPriceColl, EntryPriceRow

    entry_price_column, entry_price_row = GetEntryPriceColl(entry_candle)
    print("Window size: ", window_size)

    entry = selected_df.iloc[window_size-1-entry_price_row, entry_price_column]
    profit = round(selected_df.iloc[window_size-1, 10+mover], 2)
    real_profit = round((budget / entry)*profit, 2)

    print(
        f"Period: {selected_df.iloc[0,9+mover]} - {selected_df.iloc[window_size-2,9+mover]}")
    print("\nBudget: ", budget)
    print("\nEntry price: ", round(entry, 2))
    print("Label (target): ", round(selected_df.iloc[window_size-1, 7], 2))
    print("Model prediction: ", round(selected_df.iloc[window_size-1, 8], 2))
    print(
        f"Market Change: {round(selected_df.iloc[window_size-1, 10+mover], 2)} $")
    print(f"Profit: {real_profit} $")

    if sentiment == True:
        sentiment_data = [mpf.make_addplot(
            sentim, type='bar', markersize=200, marker='v', panel=1)]

        mpf.plot(quotes, addplot=sentiment_data, type='candle', style='starsandstripes',
                 alines=dict(alines=[datepairs_ema6, datepairs_ema12, datepairs_ema24], colors=['r', 'g', 'b']), panel_ratios=(1, 0.25),
                 figscale=1.5)  # datepairs_ema12,datepairs_ema24 ,figratio=(1,1)
    else:
        mpf.plot(quotes, type='candle', style='starsandstripes',
                 alines=dict(alines=[datepairs_ema6, datepairs_ema12, datepairs_ema24], colors=[
                             'r', 'g', 'b']),
                 figscale=1.5)  # datepairs_ema12,datepairs_ema24 ,figratio=(1,1)
    return selected_df


def PlotCurrentFormation(trade_formation, sentiment):

    selected_df = trade_formation.reset_index()

    if sentiment == True:
        sentim = selected_df.iloc[:, -1]

    # Get EMAs
    dates = selected_df['Date']

    # Get EMAs
    dates = selected_df['Date']
    ema6 = selected_df['EMA6']
    ema12 = selected_df['EMA12']
    ema24 = selected_df['EMA24']

    datepairs_ema6 = [(d1, d2) for d1, d2 in zip(dates, ema6)]
    datepairs_ema12 = [(d1, d2) for d1, d2 in zip(dates, ema12)]
    datepairs_ema24 = [(d1, d2) for d1, d2 in zip(dates, ema24)]

    # Format Dataframe
    quotes = selected_df.iloc[:, :10]
    quotes['Date'] = quotes['Date'].astype('datetime64')
    quotes = quotes.set_index('Date')

    try:
        quotes = quotes.drop('index', axis=1)
    except:
        pass

    # Plot
    quotes = quotes.iloc[:, :7]
    # print(quotes)
    quotes.columns = ['open', 'high', 'low', 'close', 'ema6', 'ema12', 'ema24']

    # Plot Chart
    if sentiment == False:
        mpf.plot(quotes, type='candle', figscale=1.5, alines=dict(alines=[
            datepairs_ema6, datepairs_ema12, datepairs_ema24], colors=['r', 'g', 'b']))
    else:
        sentiment_data = [mpf.make_addplot(
            sentim, type='bar', markersize=200, marker='v', panel=1)]

        mpf.plot(quotes, addplot=sentiment_data, type='candle', figscale=1.5, panel_ratios=(1, 0.25), alines=dict(alines=[
            datepairs_ema6, datepairs_ema12, datepairs_ema24], colors=['r', 'g', 'b']))


# coding: utf-8

# Imports and Function Definitions

from util import *
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats
import itertools

import QLearner as ql


def fill_df(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


# In[85]:

def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices,
    allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """
    norm_prices = prices / prices.ix[0]
    # scale_vec = np.dot(norm_prices, allocs)
    scale_vec = norm_prices * allocs
    port_val = scale_vec * start_val
    return port_val.sum(axis=1)


# In[86]:

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # TODO: Your code here
    daily_ret = (port_val / port_val.shift(1) - 1).ix[1:]

    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()

    cum_ret = (port_val.iloc[-1] / port_val.ix[0]) - 1

    sharpe_ratio = (daily_ret - daily_rf).mean() / (daily_ret - daily_rf).std()

    k = np.sqrt(samples_per_year)

    sharpe_ratio = k * sharpe_ratio

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


# In[87]:

def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000.,
                     leverage_sensitive=True):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    start_val = start_val * 1.0
    if not isinstance(orders_file, pd.DataFrame):
        order_df = pd.read_csv(orders_file, parse_dates=True)
    else:
        order_df = orders_file.copy()
    order_df.sort_values('Date', inplace=True)
    order_df.Date = pd.to_datetime(order_df.Date)

    start_date = order_df.Date.min()
    end_date = order_df.Date.max()  # + dt.timedelta(days=1)
    syms = list(order_df.Symbol.unique())
    prices = get_data(syms, pd.date_range(start_date, end_date))

    order_df['Shares_final'] = order_df.apply(
        lambda x: x.Shares if x.Order == 'BUY' else -x.Shares,
        axis=1
    )

    trades = pd.DataFrame(order_df.pivot_table(
        index='Date',
        columns='Symbol',
        values='Shares_final',
        fill_value=0,
        aggfunc=sum
    ))

    def get_portvals(trade_df):
        # get dollar value of each trade
        # prices multiplied by -1 because a buy subtracts cash and sell adds
        trade_df = trade_df.copy()
        trade_dollars = trade_df.multiply(-prices.ix[trade_df.index, syms])

        cash_ts = pd.Series(data=0, index=prices.index)
        cash_ts.ix[0] = start_val

        cash_ts = cash_ts.combine(
            trade_dollars.sum(1),
            func=lambda x, y: x + y,
            fill_value=0
        ).cumsum()

        trade_df['_CASH'] = cash_ts

        holdings = pd.DataFrame(data={'_CASH': cash_ts}).merge(
            trade_df[syms],
            how='left',
            left_index=True,
            right_index=True
        )

        holdings[syms] = holdings[syms].fillna(0).cumsum()

        prices['_CASH'] = 1.0

        portvals = holdings[syms].multiply(prices[syms]).sum(1) + cash_ts

        leverage = holdings[syms].multiply(
            prices[syms]).abs().sum(axis=1) / portvals

        trade_df['leverage'] = leverage

        return trade_df, portvals

    trades, portvals = get_portvals(trades)

    if leverage_sensitive:
        while (trades.leverage > 2.0).max():
            first_breach = trades[trades.leverage > 2.0].index.min()
            trades.ix[first_breach, syms] = 0

            trades, portvals = get_portvals(trades[syms])

    return pd.DataFrame(portvals)


# In[88]:

def get_bol_bands(price_df, sym, n=20):
    sma = pd.rolling_mean(price_df[sym], n)
    sp_std = pd.rolling_std(price_df[sym], n)
    ub = sma + (2 * sp_std)
    lb = sma - (2 * sp_std)
    return ub, lb, sma


# In[89]:

def plot_normalized_data(df, title="Normalized prices",
                         xlabel="Date", ylabel="Normalized price",
                         save=False, show=True):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    df_p = df / df.ix[0]
    ax = df_p.plot(title=title, fontsize=12)
    #ax.set_xlabel = xlabel
    #ax.set_ylabel = ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig('./output/plot.png')
    if show:
        plt.show()
#     plot_data(df_p, title, xlabel, ylabel)


# In[90]:

def make_order(ord_type, date, sym, nshares=100):
    tmp = pd.DataFrame({
        'Date': [date],
        'Order': [ord_type],
        'Symbol': [sym],
        'Shares': [nshares]
    })
    return tmp


def add_order(df, order):
    l_df = df.copy()
    return l_df.append(order, ignore_index=True)


# In[91]:

def bb_val(df, sym, n):
    return (df[sym] - pd.rolling_mean(df[sym], n)) / (2 * pd.rolling_std(df[sym], n))


# def bb_val(df, sym, n):
#     return (df[sym] - pd.ewma(df[sym], span=n)) / (2 * pd.ewmstd(df[sym], span=n))


# In[92]:

def rsi(df, sym, n=14):
    ''' rsi indicator '''
    gain = (df[sym] - df[sym].shift(1)).fillna(
        0)  # calculate price gain with previous day, first row nan is filled with 0

    def rsiCalc(p):
    # subfunction for calculating rsi for one lookback period
        avgGain = p[p > 0].sum() / n
        avgLoss = -p[p < 0].sum() / n
        rs = avgGain / avgLoss
        return 100 - 100 / (1 + rs)

    # run for all periods with rolling_apply
    return pd.rolling_apply(gain, n, rsiCalc)


class StrategyLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.n_episodes = 21
        self.input_scalers = None
        np.random.seed(420)
        return

    # build data
    def build_data(self, symbol, sd, ed):

        df = fill_df(get_data([symbol], pd.date_range(sd, ed)))

#         for t in range(3, 10, 2) + range(10, 26, 10):
        # for t in [10]:
        for t in range(5, 11, 5):
            df['mom{}'.format(t)] = (df[symbol] / df[symbol].shift(t)) - 1
            df['mom{}_lag'.format(t)] = df['mom{}'.format(t)].shift(1)
            df['ret_vol{}'.format(t)] = pd.rolling_std((df[symbol] / df[symbol].shift(1)) - 1, t)
            df['bb_value{}'.format(t)] = bb_val(df, symbol, t)
    #         df['bb{}_lag'.format(t)] = df['bb_value{}'.format(t)].shift(1)
    #         df['mom{}_diff'.format(t)] = df['mom{}'.format(t)] - ((df.SPY / df.SPY.shift(t)) - 1)
            df['rsi{}'.format(t)] = rsi(df, symbol, t)
            # if symbol != 'ML4T-220':
            #     df['bb{}_diff'.format(t)] = df['bb_value{}'.format(t)] - bb_val(df, 'SPY', t)



        df = df.fillna(method='bfill')

        x_cols = [c for c in df.columns if c not in ['SPY', symbol, 'y']]

        xs = df[x_cols]

        if not self.input_scalers:
            self.input_scalers = xs.mean(), xs.std()

        xs = ((xs - self.input_scalers[0]) / self.input_scalers[-1])

        return df, xs

    def smash(self, row):
        res = ''
        for s in row:
            res += str(s)
        return long(res)

    def quantize(self, x_env):
        return x_env.apply(lambda x: pd.cut(x, self.qcutz, labels=False)
                           ).apply(self.smash, axis=1)

    def run_episodes(self,
                     n_episodes,
                     x_series,
                     price_df,
                     symbol,
                     train=True):
        df = price_df.copy()
        delta = 9999999
        lval = -1e99
        last_delta = [9e9, 0] * 10
        ep_ctr = 0

        while ep_ctr < n_episodes:
            # start with no holding
            position = 1
            action = 1
            entry_price = 0

            if ep_ctr >= n_episodes - 1:
                train = False

            if not train:
                # create DF for orders
                orders = pd.DataFrame(
                    data=df.index,
                    columns=['Date'])

            for i, state in enumerate(self.quantize(x_series).iteritems()):
                try:
                    cur_state = self.pos_states.index(
                        long(str(state[1]) + str(action)))
                except ValueError:
                    if not train:
                        orders = add_order(orders, make_order(
                            'FILLER',
                            state[0], symbol, nshares=0))
                    continue
                cur_price = df.ix[state[0], symbol]

                if position == 2:
                    reward = (cur_price - entry_price) / entry_price
                if position == 0:
                    reward = (entry_price - cur_price) / entry_price
                if position == 1:
                    reward = 0

                reward *= 100

                position = action
                if i == 0:
                    # initialize learner
                    action = self.learner.querysetstate(
                        self.pos_states.index(
                            int(str(state[1]) + str(position)))
                    )

                else:
                    action = self.learner.query(
                        self.pos_states.index(
                            int(str(state[1]) + str(position))),
                        reward
                    )
                # action != position means closing or opening new position
                if action != position:

                    # action 2 = long
                    if action == 2:
                        entry_price = cur_price
                        if not train:
                            orders = add_order(
                                orders, make_order('BUY', state[0], symbol, nshares=100))
                            # if we were previously short, have to go long in
                            # addition to closing short
                            if position == 0:
                                orders = add_order(
                                    orders, make_order('BUY', state[0], symbol, nshares=100))

                    # action 0 = short
                    if action == 0:
                        entry_price = cur_price
                        if not train:
                            orders = add_order(
                                orders, make_order('SELL', state[0], symbol, nshares=100))
                            if position == 2:
                                orders = add_order(
                                    orders, make_order('SELL', state[0], symbol, nshares=100))

                    # action 1 = neutral/no position
                    # so sell to close a long or buy to close a short
                    if action == 1:
                        if not train:
                            orders = add_order(orders, make_order(
                                'SELL' if position == 2 else 'BUY',
                                state[0], symbol, nshares=100))

                if (action == position) & (not train):
                    orders = add_order(orders, make_order(
                        'FILLER',
                        state[0], symbol, nshares=0))

            if not train:
                orders = orders.dropna()

            if (self.verbose) & (ep_ctr % 5 == 0) & (not train):
                portvals = compute_portvals(
                    orders, 10000, leverage_sensitive=False)

                cval = portvals.ix[-1, 0]
                delta = abs(lval - cval) / lval
                last_delta[ep_ctr % 20] = delta
            if (self.verbose) & (ep_ctr % 5 == 0) & (not train):
                print cval, 'EP: {}, RAR: {}, ALPHA: {} |'.format(ep_ctr, np.round(self.learner.rar, 3), np.round(self.learner.alpha, 3)),
                lval = cval

            if ep_ctr > 5:
                self.learner.alpha *= (1 - (1.0 * ep_ctr / n_episodes) ** 2)

            ep_ctr += 1

            if max(last_delta) - min(last_delta) < .01:
                break

        if self.verbose:
            print 'Converged @ {}'.format(np.round(lval))
        return orders if not train else None

    def addEvidence(self,
                    symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):

        self.train_prices, self.train_x = self.build_data(symbol, sd, ed)

        self.qcutz = stats.norm.ppf(np.arange(0, 1.001, 1 / 8.0))

        self.pos_states = map(
            self.smash, itertools.product(self.quantize(self.train_x), [0, 1, 2]))

        self.learner = ql.QLearner(
            num_states=len(self.pos_states),
            num_actions=3,
            dyna=400,
            rar=0,
            radr=.999,
            alpha=.999,
            gamma=.9
        )
        self.learner.q_table += 500.0 if symbol == 'IBM' else 10

        self.train_orders = self.run_episodes(
            n_episodes=self.n_episodes,
            x_series=self.train_x,
            price_df=self.train_prices,
            symbol=symbol,
            train=True
        )
        return

    def parse_orders(self, order_df, sd, ed):
        tmp = pd.DataFrame(order_df.set_index(keys='Date').apply(lambda x: -x.Shares if x.Order=='SELL' else x.Shares, axis=1))
        tmp_dts = pd.date_range(sd, ed)
        tmp2 = get_data(['SPY'],dates=tmp_dts)
        tmp = tmp.join(pd.DataFrame(index=tmp2.index), how='outer').fillna(0)
        return tmp.groupby(level=0).sum()


    def testPolicy(self,
                   symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):
        self.test_prices, self.test_x = self.build_data(
            symbol=symbol, sd=sd, ed=ed)

        self.learner.rar = 0
        self.learner.alpha = 0
        self.learner.dyna = 0
        # can just use run episodes, because alpha is 0 so Q updates won't
        # matter
        self.test_orders = self.run_episodes(
            n_episodes=1,
            x_series=self.test_x,
            price_df=self.test_prices,
            symbol=symbol,
            train=False
        )
        return self.parse_orders(self.test_orders, sd, ed)


def main():
    # In[94]:

    learner = StrategyLearner(verbose=True)  # constructor
    learner.addEvidence(symbol="IBM", sd=dt.datetime(
        2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000)  # training step
    orders = learner.testPolicy(symbol="IBM", sd=dt.datetime(
        2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000)  # testing step
    print(orders)

    orders[orders != 0]

    # In[95]:

    learner = StrategyLearner(verbose=True)  # constructor
    learner.addEvidence(symbol="IBM", sd=dt.datetime(
        2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000)  # training step
    orders = learner.testPolicy(symbol="IBM", sd=dt.datetime(
        2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000)  # testing step
    print(orders)

    orders[orders != 0]

    # In[96]:

    learner = StrategyLearner(verbose=True)  # constructor
    learner.addEvidence(symbol="IBM", sd=dt.datetime(
        2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000)  # training step
    orders = learner.testPolicy(symbol="IBM", sd=dt.datetime(
        2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000)  # testing step
    print(orders)

    orders[orders != 0]

    # In[97]:

    learner = StrategyLearner(verbose=True)  # constructor
    learner.addEvidence(symbol="IBM", sd=dt.datetime(
        2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000)  # training step
    orders = learner.testPolicy(symbol="IBM", sd=dt.datetime(
        2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000)  # testing step
    print(orders)

    orders[orders != 0]

if __name__ == '__main__':
    main()

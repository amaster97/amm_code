import numpy as np
def balancer_IL(price_0, qty_0, price_1, qty_1=None, pool_ratio=None):
    """
    Calculates IL and PnL for constant mean product of arbitrary # assets and pool weights
    IL is calculated independent of final quantity (qty_1) as that includes trading fees
    
    Requires:
    All parameters are np arrays    
    price_0: initial prices of assets
    qty_0: initial inventory
    price_1: final prices
    qty_1: final inventory; if not provided, then assumed to be theoretical calculated inventory
    pool_ratio: % equity of each asset in pool in np array; if not provided, calculated implicitly from price and qty
    
    
    Returns:
    dict containing
    IL_percent: % loss due to IL
    IL_dollar: USD loss due to IL
    delta_pnl: PnL if just held assets
    predicted_qty_1:  predicted new qty of assets based on new prices
    predicted_equity_1:  predicted equity of assets based on new prices in pool
    equity_held:  predicted equity of assets if just held
    net_pnl: total_pnl including fees as derived sum(price_1 * qty_1)
    implied_fees_pnl:  implied fees based on differencel of actual equity and implied equity
    
    Invariances:
    prod(power(qty_0, ratio)) = prod(power(qty_1, ratio))
    price_0 * qty_0 /sum(price_1 * qty_1) = price_1 * qty_1 / sum(price_1 * qty_1) = pool_ratio
    net_pnl = delta_pnl + IL_dollar + implied_fees_pnl
    IL_dollar = equity_held - predicted_equity_1
    implied_fees_pnl = net_pnl - delta_pnl - IL_dollar
    
    """
    if pool_ratio is None:
        pool_ratio = price_0 * qty_0 / (np.dot(price_0, qty_0))
    
    N = len(price_0)
    K = np.prod(np.power(qty_0, pool_ratio))
    
    ratio_div_px1 = pool_ratio / price_1
    ratio_px1_factor = np.prod(np.power(ratio_div_px1, pool_ratio))
    
    
    calc_qty_1 = K * ratio_px1_factor**-1 * ratio_div_px1
    calc_equity_1 = np.sum(price_1 * calc_qty_1)
    equity_0 = np.sum(price_0 * qty_0)
    
    if qty_1 is not None:
        equity_1 = np.sum(price_1 * qty_1)
    else:
        equity_1 = calc_equity_1
    
    equity_held = np.sum(price_1 * qty_0)
    
    IL_dollar = calc_equity_1 - equity_held
    IL_percent = IL_dollar / equity_held
    
    net_pnl = (equity_1 - equity_0)
    delta_pnl = equity_held - equity_0

    return {
        'IL_percent': IL_percent,
        'IL_dollar': IL_dollar,
        'delta_pnl': delta_pnl,
        'predicted_qty_1': calc_qty_1,
        'predicted_equity_1': calc_equity_1,
        'equity_held': equity_held,
        'net_pnl': net_pnl,
        'implied_fees_pnl': net_pnl - delta_pnl - IL_dollar,
        'pool_ratio': pool_ratio,
    
    }                               

def secant(f,a,b,N):
    '''Approximate solution of f(x)=0 on interval [a,b] by the secant method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    m_N : number
        The x intercept of the secant line on the the Nth interval
            m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
        for some intercept m_n then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iterations, the secant method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> secant(f,1,2,5)
    1.6180257510729614
    '''
    if f(a)*f(b) >= 0:
        print("Secant method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            #print("Found exact solution.")
            return m_n
        else:
            print("Secant method fails.")
            return None
    return a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))


def balancer_apr_to_move(APR, pool_ratio):
    """
    Given target APR and pool ratio of two assets, returns the percentage movement of 2nd asset vs 1st to breakeven with APR
    Uses secant method and solves numerically
    
    Requires:
    APR: Positive float of implied APR of pool
    pool_ratio:  np array of the ratio of two asssets
    
    Returns:
    dict containing
    breakeven increase %: How much 2nd asset (idx1) can increase by to breakeven with IL = APR
    breakeven decrease %: How much 2nd asset (idx1) can decrease by to breakeven with IL = APR
    IV:  Implied vol given breakeven endpoints
    
    """
    
    def f(pb1):
        return 1 - ((pb1**pool_ratio[1]) / (1-pool_ratio[1] + pool_ratio[1]*pb1)) - APR
    
    left = secant(f, min(APR/2, 1e-15), 1-min(APR/2, 1e-15), 10000)
    right = secant(f, 1+min(APR/2, 1e-15), 1e15, 10000)
    log_IV = np.std([np.log(left), np.log(right)])
        

    return {'breakeven increase %': right-1, 'breakeven decrease %': left-1, 'IV': log_IV}
    
    
def balancer_APR(fee_percent, daily_volume, TVL, compound):
    """
    Estimates the APR of a pool given trading volume, tvl, fees, and compound.  Uses simple interest if compound is False, else uses daily compound interest
    
    Requires:
    fee_percent: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    TVL:  Total Value Locked of Pool
    compound:  True if fees are automatically reinvested (i.e. Sushiswap), False if not (i.e. Uni V3) 

    Note daily_volume and TVL MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)

    Returns:
    Annual APR estimate
    """
    
    
    daily_return = fee_percent * daily_volume / TVL
    
    if compound:
        yearly_return = (1+daily_return)**365 - 1
    
    else:
        yearly_return = daily_return*365
        
    return yearly_return
    
    
def balancer_IV(fee_percent, daily_volume, TVL, compound, pool_ratio, fast=False):
    """
    Approximates annualized IV of balancer / Uni V2 / Sushi constant product pool
    https://medium.com/@danielalcarraz_42353/calculating-implied-volatility-from-uniswap-v2-v3-e466e49d60e0
    
    Requires:
    fee_percent: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    TVL:  Total Value Locked of Pool
    compound:  True if fees are automatically reinvested (i.e. Sushiswap), False if not (i.e. Uni V3) 
    pool_ratio:  The dollar weighted ratio of assets, must be exactly 2 assets
    fast: uses article formula to get a fast approximation if set true.  Works well for short timescales or small IV, else breaksdown.  Else attempts to use secant method to find more exact sol


    Note daily_volume and TVL MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)

    Returns:
    dict containing:
    Implied Vol of Pool
    APR of Pool
    """    
    

    yearly_return = balancer_APR(fee_percent, daily_volume, TVL, compound)    
    IV_article = np.sqrt(8*yearly_return)
    print('Approximation IV:', IV_article)
    
        
    if fast:
        return IV_article
    try:
        IV_Adam = balancer_apr_to_move(yearly_return, pool_ratio)['IV']
    
    except:
        IV_Adam = IV_article
        print('More precise method failed, reverting to appx. IV')
        
    print('More Precise IV:', IV_Adam)
    
    return {'IV': IV_Adam, 'APR': yearly_return}
    
def Uni_v3_L(P, p_low, p_high, qty_x=-1, qty_y=-1):
    """
    Calculates the Liquidity of a pool given the relative price, relative price bounds, and current qty
    
    Requires:
    P:  Relative price of assets, price_x / price_y
    p_low, p_high:  bounds for range
    qty_x, qty_y: at least one must be specified.  Will calculate the qty of other based on invariance factor of the AMM.
    qty_x,qty_y must also be valid given the current price P, and p_low,p_high of bounds to construct a valid LP position.  
    Will raise error if not.
    
    Returns:
    Liquidity of the pool, https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf  
    """
    
    if p_low >= p_high:
        raise ValueError('Requires p_high > p_low, given p_high {}, p_low {}'.format(p_high, p_low))
    
    #print(P, p_low, p_high, qty_x, qty_y, 'hahah')
    if P <= p_low:
        if (qty_x <= 0):
            raise ValueError('Requires positive qty_x when P <= p_low, provided value {}'.format(qty_x))
    
        return qty_x * np.sqrt(p_low * p_high) / (np.sqrt(p_high) - np.sqrt(p_low))
    
    elif P >= p_high:
        if (qty_y <= 0):
            raise ValueError('Requires positive qty_y when P >= p_high, provided value {}'.format(qty_y))
    
        return qty_y/ (np.sqrt(p_high) - np.sqrt(p_low))
        
    else:
        if (qty_x <= 0) and (qty_y <= 0):
            raise ValueError('Require either a positive qty_x or qty_y when p_low < P < p_high, provided values {}, {}'.format(qty_x, qty_y))
        
        x_liq = qty_x * np.sqrt(P * p_high) / (np.sqrt(p_high) - np.sqrt(P))
        y_liq = qty_y / (np.sqrt(P) - np.sqrt(p_low))
        
        #print(x_liq, y_liq,'x_liq, y_liq')
        
        if qty_x <= 0:
            return y_liq
        elif qty_y <= 0:
            return x_liq
        else:
            if (np.abs(x_liq - y_liq) > 1e-3 * np.sqrt(x_liq * y_liq)):
                raise ValueError('Liquidity Calculation for x and y differ by more than 1%, x_liq: {}, y_liq: {} with x_qty: {}, y_qty: {}'.format(x_liq, y_liq, qty_x, qty_y))
            return min(x_liq, y_liq)

def uni_v3_qty(P, L, p_low, p_high):
    """
    Given liquidity of pool, price, and price bounds, finds the quantity of each asset
    
    Requires:
    P:  Relative price of assets, price_x / price_y
    L:  Liquidity of pool
    p_low, p_high:  bounds for range

    Returns:
    np.array([qty_x, qty_y]) 
    """
    
    if P <= p_low:
        y = 0
        x = L * (np.sqrt(p_high) - np.sqrt(p_low)) / (np.sqrt(p_high) * np.sqrt(p_low))
    elif P >= p_high:
        x = 0
        y = L * (np.sqrt(p_high) - np.sqrt(p_low))
        
    else:
        x = L * (np.sqrt(p_high) - np.sqrt(P)) / (np.sqrt(P) * np.sqrt(p_high))
        y = L * (np.sqrt(P) - np.sqrt(p_low))
    

    #print('in qty func', x, y)
    return np.array([x, y])
    

def uni_v3_IL_calc(price_0, price_1, rel_px_range=None, qty_0_A=-1, qty_0_B=-1):
    """
    Calculates IL and PnL for Uniswap V3 pool with given liquidity ranges
    Since every LP pair is an NFT and fees are accured separately, there's no need / way to implicitly calculate fees with this
    Uniswap V3 IL Calculator
    References:
    https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf
    https://github.com/atiselsts/uniswap-v3-liquidity-math/blob/master/uniswap-v3-liquidity-math.py
    
    Requires:
    price_0: initial prices of assets as np array
    price_1: final prices of assets as np_array
    rel_px_range: Price bounds of liquidity provision denoted in terms of asset 1 in arrays (Px_Asset_idx0 / Px_Asset_idx1).  If none provided, assume range is entire curve where min and max are P_0 */ by 1e15
    qty_0_A, qty_0_B: at least one must be specified.  Will calculate the qty of other based on invariance factor of the AMM.
    Qty must also be valid given the current prices price_0 and rel_px_range of bounds to construct a valid LP position.  Will raise error if not.
    
    Returns:
    dict containing
    
    IL_percent: % loss due to IL
    IL_dollar: USD loss due to IL
    delta_pnl: PnL if just held assets
    qty_new:  predicted new qty of assets based on new prices.
    equity_new:  predicted equity of assets based on new prices in pool
    equity_held:  predicted equity of assets if just held
    net_pnl: total_pnl including fees as derived sum(price_1 * qty_1)
    
    Invariances:
    (qty[0] + L /sqrt(rel_px_range[1]))*(qty[1] + L*sqrt(rel_px_range[0])) = L^2
    net_pnl = delta_pnl + IL_dollar
    IL_dollar = equity_held - equity_new    
    """
    P_0 = price_0[0] / price_0[1]
    
    if rel_px_range is None:
        rel_px_range = np.array([P_0*1e-15, P_0*1e15])
    
    L = Uni_v3_L(P_0, rel_px_range[0], rel_px_range[1], qty_0_A, qty_0_B)
    qty_0 = uni_v3_qty(P_0, L, rel_px_range[0], rel_px_range[1])

    P_1 = price_1[0] / price_1[1]    
    qty_1 = uni_v3_qty(P_1, L, rel_px_range[0], rel_px_range[1])
    
    equity_1 = np.sum(price_1 * qty_1)
    equity_0 = np.sum(price_0 * qty_0)
    
    equity_held = np.sum(price_1 * qty_0)
    
    IL_dollar = equity_1 - equity_held
    IL_percent = IL_dollar / equity_held
    
    net_pnl = (equity_1 - equity_0)
    delta_pnl = equity_held - equity_0

    return {
        'IL_percent': IL_percent,
        'IL_dollar': IL_dollar,
        'delta_pnl': delta_pnl,
        'qty_new': qty_1,
        'equity_new': equity_1,
        'equity_held': equity_held,
        'net_pnl': net_pnl,
        'L': L,
    }                               




def uni_v3_apr_to_move(APR, price, rel_px_range=None):
    """
    Given target APR and pool ratio of two assets, returns the percentage movement of 2nd asset vs 1st to breakeven with APR
    Uses secant method and solves numerically
    
    Requires:
    APR: Positive float of implied APR of pool
    price: np array that contains the current price of two assets
    rel_px_range: relative price range of bounds in uniswap pool px_asset_0 / px_asset_1. The relative price must fall within this range.  Defaults to 0 - inf for regular xy=k pool, and will solve using balancer calculator.
    
    Returns:
    dict containing
    breakeven increase %: How much 1st asset (idx0) can increase by to breakeven with IL = APR
    breakeven decrease %: How much 1st asset (idx0) can decrease by to breakeven with IL = APR
    IV:  Implied vol given breakeven endpoints

    If the values of these are either 0 or inf, it means that the APR was so large that with the given bounds, it is impossible to breakeven with IL.
    
    """
    
        
    P = price[0] / price[1]
    
    if rel_px_range is None:
        return balancer_apr_to_move(APR, np.array([0.5, .5]))
    
    if (P <= rel_px_range[0] or P >= rel_px_range[1]):
        raise ValueError('Effective price {} is not between range of {}, with provided prices {}'.format(P, rel_px_range, price))
    """
    qty_x = 1
    a,b = rel_px_range[0], rel_px_range[1]
    L = Uni_v3_L(P, rel_px_range[0], rel_px_range[1], qty_x)
    """
    qty_x = 1
    a,b = rel_px_range[0]/P, rel_px_range[1]/P
    L = Uni_v3_L(1, a, b, qty_x)    
    def f(P1):
        root_b = np.sqrt(b)
        root_x = np.sqrt(P1)
        root_a = np.sqrt(a)
        x = P1
        return 1 - APR - (x * L * (root_b - root_x) / (root_b * root_x) + L*(root_x - root_a)) / (x + L*(1 - root_a))
        
    left = secant(f, min(APR/2, 1e-15), (1-min(APR/2, 1e-15)), 10000)
    right = secant(f, (1+min(APR/2, 1e-15)), max(APR, 1e15), 10000)
        
    if left == None:
        left = 0
    if right == None:
        right = float('inf')
    
    left_p = left *P
    right_p = right *P
    
    if left_p < rel_px_range[0]:
        print('Price drop requirement more than lower range bound {} < {}'.format(left_p, rel_px_range[0]))
        left -= 1
    else:
        left -=1
    if right_p > rel_px_range[1]:
        print('Price increase requirement more than upper range bound {} > {}'.format(right_p, rel_px_range[1]))
        right -=1
    else:
        right -= 1
        
    if (left is not None) and (right is not None):
        vol = np.std([np.log(left+1), np.log(right+1)])
    else:
        vol = None
        
    return {'breakeven increase %': right, 'breakeven decrease %': left, 'IV': vol}
    
    
def uni_v3_IV_tight(fee_percent, daily_volume, atm_tick_liq):
    """
    Approximates the annualized IV of a Uni_v3 pool
    Works best for when the price bounds are not very wide (ratio of high/low price bounds < 2)
    https://lambert-guillaume.medium.com/on-chain-volatility-and-uniswap-v3-d031b98143d1
    
    Requires:
    fee_percent: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    atm_tick_liq:  Liquidity present at the atm tick
    
    Note daily_volume and atm_tick_liq MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)
    
    Returns:
    Implied Vol of current tick
        
    """
    return 2 * fee_percent * np.sqrt(daily_volume * 365 / atm_tick_liq)

    
    
def uni_v3_IV_wide(fee_percent, daily_volume, atm_tick_liq, prices_arr, price_range_arr, total_ticks, yearly_hedge_freq):
    """
    Approximates the annualized IV of a Uni_v3 pool  when the price bounds are wide (ratio of high/low price bounds >= 2)
    Uses an exponential weighted average of uni_v3_tight and balancer_IV with mean ~ 2
    w_range = 2/ (1 + exp(-(P-1))) - 1, w_tick = 1 - w_range
    
    Requires:
    fee_percent: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    atm_tick_liq:  Liquidity present at the atm tick
    prices_arr:  current prices of assets
    prices_range_arr: lower, higher price bounds of assets Px_Asset_idx0 / Px_Asset_idx0
    total_ticks:  total number of distinct liquidity ticks within provided range, integer
    yearly_hedge_freq:  Theoretically represents the rough frequency that we hedge deltas during the year.  Practically increase this number if the APR is big / the price range is tight s.t. calculating IL for a given APR will force the price outside of the range.

    Note daily_volume, atm_tick_liq,  MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)
    
    Returns:
    Implied Vol of range
        
    """
    
    tick_IV = uni_v3_IV_tight(fee_percent, daily_volume, atm_tick_liq)
    
    ## calculate adjusted liq given the liquidity of atm vs the average liquidity of pool
    adj_volume = daily_volume / total_ticks
    
    pool_ratio = uni_v3_qty(prices_arr[0]/prices_arr[1], 1, price_range_arr[0], price_range_arr[1])
    pool_ratio *= prices_arr
    pool_ratio /= np.sum(pool_ratio)
    
    price_bnds_ratio = price_range_arr[1] / price_range_arr[0]
    
    
    
    range_apr = balancer_APR(fee_percent, adj_volume, atm_tick_liq, False)
    
    print('range apr', range_apr)
    
    adj_apr = (1+range_apr)**(1/yearly_hedge_freq) - 1
    
    adj_apr = range_apr/yearly_hedge_freq
    
    range_IV = uni_v3_apr_to_move(adj_apr, prices_arr, price_range_arr)['IV'] * np.sqrt(yearly_hedge_freq)

    w_range = 2 / (1+np.exp(-(price_bnds_ratio-1))) - 1
    w_tick = 1 - w_range
    print('w_range', w_range)
    print('w_tick', w_tick)
    print(tick_IV, 'tick iv')
    print(range_IV, 'range iv')
    
    var_f = (w_range)*(range_IV)**2 + (w_tick)*(tick_IV)**2
    vol_f = np.sqrt(var_f)
    print(vol_f, 'volf')
    return vol_f
    
    
def gamma_appx_balancer(prices_arr, ratios_arr, eps=1e-5):
    """
    Numerically estimates the gamma of assets given a balancer style pool with current prices and pool ratio.
    
    Requires:
    prices_arr: current prices of assets as array
    ratios_arr: % of each ratio in pool
    eps: pertubation % of price for each asset to derive gamma approximation, defaults 1e-5
    
    
    Returns:
    array containing the gamma estimation per unit asset currently in pool
    """
    
    prices_arr = np.array(prices_arr, dtype=np.float64)
    qty_0 = ratios_arr / prices_arr
    gammas = []
    
    for i in range(len(prices_arr)):
        
        curr_asset_px = prices_arr[i]
        cur_asset_low = curr_asset_px * (1-eps)
        cur_asset_high = curr_asset_px * (1+eps)
        
        change_asset_price = cur_asset_high - cur_asset_low
        
        px_low = np.array(prices_arr)
        px_high = np.array(prices_arr)
        px_low[i] = cur_asset_low
        px_high[i] = cur_asset_high
        
        
        qty = qty_0 / qty_0[i]
        
        
        deltas_low = balancer_IL(prices_arr, qty, px_low)['predicted_qty_1'][i]
        deltas_high = balancer_IL(prices_arr, qty, px_high)['predicted_qty_1'][i]
        
        deltas_diff = deltas_high - deltas_low
        gamma = deltas_diff / change_asset_price
        
        gammas.append(gamma)
    
    return gammas
    
    
def gamma_appx_univ3(prices_arr, rel_px_range, eps=1e-5):
    """
    Numerically estimates the unit gamma of assets given a uniswap v3 style pool with current prices and price bounds.
    
    Requires:
    prices_arr: current prices of assets in array
    rel_px_range:  the low and high price bounds denoted in terms of asset 1 (Px_Asset_idx0 / Px_Asset_idx1)
    eps: pertubation % of price for each asset to derive gamma approximation, defaults 1e-5
    
    
    Returns:
    array containing the gamma estimation per unit asset currently in pool
    """
    
    prices_arr = np.array(prices_arr, dtype=np.float64)
    gammas = []
    
    for i in range(len(prices_arr)):
        
        curr_asset_px = prices_arr[i]
        cur_asset_low = curr_asset_px * (1-eps)
        cur_asset_high = curr_asset_px * (1+eps)
        
        change_asset_price = cur_asset_high - cur_asset_low
        
        px_low = np.array(prices_arr)
        px_high = np.array(prices_arr)
        px_low[i] = cur_asset_low
        px_high[i] = cur_asset_high
        
            
        if i == 0:
            deltas_low = uni_v3_IL_calc(prices_arr, px_low, rel_px_range, qty_0_A = 1)['qty_new'][i]
            deltas_high = uni_v3_IL_calc(prices_arr, px_high, rel_px_range, qty_0_A = 1)['qty_new'][i]
        
        else:
            deltas_low = uni_v3_IL_calc(prices_arr, px_low, rel_px_range, qty_0_B = 1)['qty_new'][i]
            deltas_high = uni_v3_IL_calc(prices_arr, px_high, rel_px_range, qty_0_B = 1)['qty_new'][i]
        
        
        deltas_diff = deltas_high - deltas_low
        gamma = deltas_diff / change_asset_price
        
        gammas.append(gamma)
    
    return gammas
    
    
def balancer_pool_returns(fee_tier, daily_volume, tvl, rewards_apr, funding_arr, ratios_arr, compound, fast=False):
    """
    Convenience function to calculate dirty/adj apr/vol given parameters of balancer style pool
    
    Requires:
    fee_tier: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    tvl:  Total Value Locked of Pool
    rewards_apr: APR of rewards on top of fees
    funding_arr: np array of perp/funding rate for assets
    ratios_arr: np array of value weight of assets
    compound:  True if fees are automatically reinvested (i.e. Sushiswap), False if not (i.e. Uni V3) 
    fast: uses a fast approximation vol if set true.  Works well for short timescales or small IV, else breaksdown.  Else attempts to use secant method to find more exact sol

    Note daily_volume and TVL MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)

    Returns:
    dict containing:
    dirty_vol: IV of pool only accounting for fees
    dirty_apr: Implied APR of dirty_vol to IL
    adj_vol: IV of pool fees + funding + rewards
    adj_apr: Implied APR of adj_vol to IL
    """        
    try:
        res = balancer_IV(fee_tier, daily_volume, tvl, compound, ratios_arr, fast)
        dirty_vol = res['IV']
        dirty_apr = res['APR']
   
    except:
        dirty_apr = balancer_APR(fee_tier, daily_volume, tvl, compound)
        dirty_vol = float('inf')
    
    adj_apr = dirty_apr + np.dot(funding_arr, ratios_arr) + rewards_apr
    
    try:
        adj_vol = balancer_apr_to_move(adj_apr, ratios_arr)['IV']
    
    except:
        adj_vol = np.nan
        
        
    return {
        'dirty_vol': dirty_vol,
        'dirty_apr': dirty_apr,
        'adj_vol': adj_vol,
        'adj_apr': adj_apr
    
    }


def uni_v3_pool_returns(fee_tier, daily_volume, atm_tick_liq, rewards_apr, funding_arr, cur_prices_arr, price_bounds_arr, compound, fast=False, num_ticks_in_range=None, int_ticks_spacing=None, yearly_hedge_freq=365):
    
    """
    Convenience function to calculate dirty/adj apr/vol given parameters of uni v3 style pool
    
    Requires:
    fee_tier: fee tier of pool
    daily_volume:  Average daily volume of pool.  If normalized over a different period, obviously adjust accordingly.
    tvl:  Total Value Locked of Pool
    rewards_apr: APR of rewards on top of fees
    funding_arr: np array of perp/funding rate for assets
    cur_prices_arr:  np array of initial asset prices
    price_bounds_arr:  np array of the price bands px_Asset_idx0 / px_Asset_idx1
    compound:  True if fees are automatically reinvested (i.e. Sushiswap), False if not (i.e. Uni V3) 
    fast: uses a fast approximation vol if set true.  Works well for short timescales or small IV, else breaksdown.  Else attempts to use secant method to find more exact sol
    num_ticks_in_range:  Total number of distinct ticks within provided liquidity band. If none, int_ticks_spacing must be present to calculate.
    int_ticks_spacing:  Difference in integers i,j between consecutive ticks s.t. i*(1.0001)^int_ticks_spacing = j as discussed in uniswap whitepaper
    yearly_hedge_freq:  Theoretically represents the rough frequency that we hedge deltas during the year.  Practically increase this number if the APR is big / the price range is tight s.t. calculating IL for a given APR will force the price outside of the range.
    

    Note daily_volume and TVL MUST be in the same unit of base currency (i.e. dollars or normalized qty_x*qty_y = L^2)

    Returns:
    dict containing:
    dirty_vol: IV of pool only accounting for fees
    dirty_apr: Implied APR of dirty_vol to IL
    adj_vol: IV of pool fees + funding + rewards
    adj_apr: Implied APR of adj_vol to IL
    
    ratios_arr:  array of % of capital in each asset
    """
    if num_ticks_in_range is None:
        if int_ticks_spacing is None:
            raise ValueError('Either num_ticks_in_range or int_ticks_spacing must be defined')
        ticks = np.log(price_bounds_arr) / np.log(1.0001)
        num_ticks_in_range = (ticks[1] - ticks[0]) / int_ticks_spacing
    
    px_bnds_ratio = price_bounds_arr[1] / price_bounds_arr[0]

    ## Get initial price ratio and normalize
    ratios_arr = uni_v3_qty(cur_prices_arr[0] / cur_prices_arr[1], 1.0, price_bounds_arr[0], price_bounds_arr[1]) * cur_prices_arr
    ratios_arr /= np.sum(ratios_arr)
    
    
    try:
        dirty_vol = uni_v3_IV_wide(fee_tier, daily_volume, atm_tick_liq, cur_prices_arr, price_bounds_arr, num_ticks_in_range, yearly_hedge_freq)
        print('dirty vol', dirty_vol)
    except:
        
        print('Secant failed and cannot get iv. Consider increasing yearly_hedge_freq parameter.')
        dirty_vol = float('inf')
        
    try:
        new_prices = np.array([np.exp(dirty_vol/np.sqrt(yearly_hedge_freq)) * cur_prices_arr[0], cur_prices_arr[1]])
        
        dirty_apr = -1*uni_v3_IL_calc(cur_prices_arr, new_prices, price_bounds_arr, 1)['IL_percent'] * yearly_hedge_freq
        
        print('new px', new_prices)
        print('dirty apr', dirty_apr)
        
        
    except:
        print('Could not get dirty_apr.  Attempting to rebuild it')
        tight_iv = uni_v3_IV_tight(fee_tier, daily_volume, atm_tick_liq)
        new_prices = np.array([np.exp(tight_iv) * cur_prices_arr[0], cur_prices_arr[1]])
        
        tight_apr = -1*uni_v3_IL_calc(cur_prices_arr, new_prices, price_bounds_arr, 1)['IL_percent']
        bal_apr = balancer_APR(fee_tier, daily_volume, range_liq, compound)
        
        w_range = 2 / (1+np.exp(-(px_bnds_ratio-1))) - 1
        w_tick = 1 - w_range
        
        dirty_apr = w_range*bal_apr + w_tick*tight_apr
        
    #print(dirty_apr, funding_arr, ratios_arr, rewards_apr)
    adj_apr = dirty_apr + np.dot(funding_arr, ratios_arr) + rewards_apr

    if adj_apr <= 0.:
        print('adj apr less than 0, iv is neg')
        adj_vol = np.nan
    
    else:

        try:
            #print('adj_apr_final', adj_apr)
            adj_vol = uni_v3_apr_to_move(adj_apr/yearly_hedge_freq, cur_prices_arr, price_bounds_arr)['IV'] * np.sqrt(yearly_hedge_freq)
            print('adj vol', adj_vol)

        except:
            print('Failed calculating vol with given apr and bounds.  Please increase yearly_hedge_freq parameter and try again.')
            adj_vol = np.nan

        
    return {
        'dirty_vol': dirty_vol,
        'dirty_apr': dirty_apr,
        'adj_vol': adj_vol,
        'adj_apr': adj_apr,
        'ratios_arr': ratios_arr,
    
    }
    
    
    
"""
EXAMPLE CODE of how i would update certain values given parameters we can pull from the pool

for f in uni_sheets:
    df = pd.read_excel(all_sheets, f)
    for idx, row in df.iterrows():
        print(row['asset0'], row['asset1'])
        #funding_arr = np.array([row['funding0'], row['funding1']])
        funding_arr = np.array([0., 0.])
        cur_px = np.array([row['px_0'], row['px_1']])
        px_bounds = np.array([row['lower_bound'], row['upper_bound']])
        res = uni_v3_pool_returns(row['fee_tier (net of protocol cut)'], row['daily_volume'], row['atm_tick_liq'], row['rewards_apr'], funding_arr, cur_px, px_bounds, False, fast=False, int_ticks_spacing=row['tick_size'])

        print(res)
        df.at[idx, 'dirty_vol'] = res['dirty_vol']
        df.at[idx, 'dirty_apr'] = res['dirty_apr']
        df.at[idx, 'adj_apr'] = res['adj_apr']
        df.at[idx, 'adj_vol'] = res['adj_vol']
        df.at[idx, 'LP_%0'] = res['ratios_arr'][0]
        df.at[idx, 'LP_%1'] = res['ratios_arr'][1]
                            
    display(df.T)
        
        


"""


"""
px_0 = np.array([1718.59, 1])
px_1 = np.array([1922, 1])
price_range = np.array([949.62, 3449.5])
qty0A = 37.99
qty0B = 56980

start = dt.date(year=2022, month=7, day =29)
end = dt.date(year = 2022, month=8, day=12)

years = (end - start).days / 365
res = uni_v3_IL_calc(px_0, px_1, price_range, qty0A, qty0B)
equity_held = res['equity_held']
equity_new = res['equity_new']
actual_equity = 130682.2552
hedge_freq = 365
#print(res)

print('ETH-USDC')
IL_loss = equity_held - equity_new
print('IL', IL_loss)
Fees_gen = actual_equity - equity_new
print('Fees', Fees_gen)
final_pnl = actual_equity - equity_held
print('PnL', final_pnl)
RV_period = abs(np.log((px_1[0] / px_1[1]) / (px_0[0] / px_0[1]))) / np.sqrt(years)
print('RV entire period', RV_period)

yearly_fees = Fees_gen / years
APR = yearly_fees / equity_held
IV_period = uni_v3_apr_to_move(APR/hedge_freq, px_0, price_range)['IV'] * np.sqrt(hedge_freq)
print('IV period', IV_period)

"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve
idx=pd.IndexSlice
from datetime import datetime


def tot_cash_cost_cal(price_dollar, minesite_cost_pto, og_percent, recovery_percent, paid_percent, tcrc_cpp, freight_cpp, royalty_cpp, ore_prod=1):
    # Total cash margin is invariant with ore_prod
    tonne_to_lbs=2204.62 # All costs are in dollars/tonne
    tot_minesite_cost = ore_prod * minesite_cost_pto
    paid_metal_prod = ore_prod * og_percent/(1e2) * recovery_percent/(1e2) * paid_percent/(1e2)
    tot_tcrc = paid_metal_prod * tcrc_cpp * tonne_to_lbs/(1e2)
    tot_freight = paid_metal_prod * freight_cpp * tonne_to_lbs/(1e2)
    tot_royalty = paid_metal_prod * royalty_cpp * tonne_to_lbs/(1e2)
    cash_costs = pd.Series([tot_minesite_cost/paid_metal_prod, tot_tcrc/paid_metal_prod, tot_freight/paid_metal_prod, tot_royalty/paid_metal_prod], 
                           index=['Minsite', 'TCRC', 'Frieght', 'Royalty'])
    return cash_costs


def tot_cash_margin_cal(price_dollar, minesite_cost_pto, og_percent, recovery_percent, paid_percent, tcrc_cpp, freight_cpp, royalty_cpp, ore_prod=1):
    # Total cash margin is invariant with ore_prod
    tonne_to_lbs=2204.62 # All costs are in dollars/tonne
    tot_minesite_cost = ore_prod * minesite_cost_pto
    paid_metal_prod = ore_prod * og_percent/(1e2) * recovery_percent/(1e2) * paid_percent/(1e2)
    tot_tcrc = paid_metal_prod * tcrc_cpp * tonne_to_lbs/(1e2)
    tot_freight = paid_metal_prod * freight_cpp * tonne_to_lbs/(1e2)
    tot_royalty = paid_metal_prod * royalty_cpp * tonne_to_lbs/(1e2)
    tot_cash_cost = tot_minesite_cost + tot_tcrc + tot_freight + tot_royalty
    tot_cash_margin = price_dollar-tot_cash_cost/paid_metal_prod
    return tot_cash_margin


def ore_prod_calculator(cap, cap_uti_base, tot_cash_margin, tcm_base, prod_elas, ramp_flag):
    if ramp_flag:
        cap_uti=0.4
    
    elif tot_cash_margin < 0:
        cap_uti=0.75
        
    else:
        cap_uti=cap_uti_base * (tot_cash_margin/tcm_base)**prod_elas
    
    ore_prod = cap * cap_uti
    return ore_prod


###### Simulate whole mine life, give whole price_series (not dynamic) ######
def simulate_mine_life(simulation_time, price_series, tcrc_cpp_series, 
                       hyper_param, mine_data, og_elas, 
                       close_price_method='max', close_years_back=5):
    # Read mine specific data
    tonne_to_lbs = 2204.62
    ore_cap_kt = mine_data.loc['Ore capacity (kt)']
    cum_ore_prod_kt = mine_data.loc['Cumulative ore treated (kt)']
    og_current_percent = mine_data.loc['Head grade (%)']
    recovery_percent = mine_data.loc['Recovery rate (%)']
    paid_percent = mine_data.loc['Payable percent (%)']

    minesite_cost_cpp = mine_data.loc['Minesite cost (cents/lb)']
    freight_cpp = mine_data.loc['Transport and offsite (cents/lb)']
    royalty_cpp = mine_data.loc['Royalty (cents/lb)']
    overhead_dollar_annual = mine_data.loc['Overhead cost ($, annual)']
    sustain_dollar_annual = mine_data.loc['Sustaining capex ($, annual)']
    reclamation_dollar = mine_data.loc['Reclamation ($)']
    cash_flow_2018_dollar = mine_data.loc['Cash flow 2018 ($)']
    
    # For minesite, the constant cost is based on per tonne of ore rather than metal
    minesite_cost_pto = minesite_cost_cpp*tonne_to_lbs/(1e2)*og_current_percent/(1e2)*recovery_percent/(1e2)*paid_percent/(1e2)
    
    # Read Hyper-parameters
    cap_uti_base = hyper_param.loc['Capacity utilization constant', 'Value']
    tcm_base = hyper_param.loc['Total cash margin constant ($/tonne)', 'Value']
    prod_elas = hyper_param.loc['Production elasticity', 'Value']
    discount = hyper_param.loc['Discount rate', 'Value']
    
    # Initiation
    n_years = simulation_time.shape[0]
    mine_life_stats = pd.DataFrame(index=simulation_time, columns=['Head grade (%)', 'Ore treated (kt)', 
                                                                   'Recovered metal production (kt)',
                                                                   'Paid metal production (kt)', 
                                                                   'Total cash margin ($/tonne paid metal)', 
                                                                   'Pre-tax cash flow ($)'])
    if cum_ore_prod_kt == 0:
        og_init_percent = og_current_percent
    else:
        og_init_percent = og_current_percent/(cum_ore_prod_kt/ore_cap_kt)**og_elas
    
    # Simulate over life of mine, year i=0 is 2018
    for i in range(n_years):
        t = simulation_time[i]
        price_t = price_series.loc[t]
        if t == datetime(2018,1,1):
            tcrc_cpp = mine_data.loc['TCRC (cents/lb)']
        else:
            tcrc_cpp = tcrc_cpp_series[t]
        
        # If no ore production before, then this first year is a ramp up. Otherwise, if cashflow<0, assume this is ramp down.
        if cum_ore_prod_kt == 0:
            ramp_up_flag=True
            ramp_down_flag=False
        elif cash_flow_2018_dollar < 0 and i == 0:
            ramp_up_flag=False
            ramp_down_flag=True
        else:
            ramp_up_flag=False
            ramp_down_flag=False
                
        # Calculate ore grade
        if i == 0:
            og_percent = og_current_percent
        else:
            og_percent = og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.84)**og_elas 
            # 0.84 is the first order approx of last year cap uti

        
        # Calculate total cash margin and cash flow under the normal scenario
        ramp_flag=ramp_up_flag or ramp_down_flag
        tot_cash_margin = tot_cash_margin_cal(price_t, minesite_cost_pto, og_percent, recovery_percent, paid_percent, 
                                              tcrc_cpp, freight_cpp, royalty_cpp)
        ore_prod_kt=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin, tcm_base, prod_elas, ramp_flag)
        recovered_metal_prod_kt = ore_prod_kt * og_percent * recovery_percent / (1e4)
        paid_metal_prod_kt = recovered_metal_prod_kt * paid_percent / (1e2)
        cash_flow_dollar = tot_cash_margin * paid_metal_prod_kt * (1e3) - overhead_dollar_annual - sustain_dollar_annual
        # Reminder: it is possible to have a negative cash flow during ramp up

        mine_life_stats.loc[t, 'Head grade (%)'] = og_percent
        mine_life_stats.loc[t, 'Total cash margin ($/tonne paid metal)'] = tot_cash_margin


        # If closing already determined at this point (ramp_down_flag=True), execute closing produciton
        if ramp_down_flag:
            # This year at 40% cap uti
            mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt
            mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt
            mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt
            mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar
            
            # Next year incurring reclamation
            t_plus_1=t+relativedelta(years=1)
            mine_life_stats.loc[t_plus_1, 'Ore treated (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Paid metal production (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Recovered metal production (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Pre-tax cash flow ($)'] = -reclamation_dollar
            break
            
        # If not closing not determined yet, calculate cashflow under expected price, not assuming ramp down yet. If this cash
        # flow turns out to be negative, begin to calculate the ramp down cash flow
        else:
            # Expected price: assume miners look back on historical price as well, use trailing price (3 years trailing for now)
            if close_price_method=='mean':
                price_expect = price_series.rolling(close_years_back).mean().loc[t]
            elif close_price_method=='max':
                t_back=t-relativedelta(years=close_years_back-1)
                price_expect = price_series.loc[t_back:t].max()
                
            tot_cash_margin_expect = tot_cash_margin_cal(price_expect, minesite_cost_pto, og_percent, recovery_percent, 
                                                         paid_percent, tcrc_cpp, freight_cpp, royalty_cpp)            
            ore_prod_kt_expect=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, tcm_base, prod_elas, 
                                                   ramp_flag=False)
            recovered_metal_prod_kt_expect = ore_prod_kt_expect * og_percent * recovery_percent / (1e4)
            paid_metal_prod_kt_expect = recovered_metal_prod_kt_expect * paid_percent / (1e2)
            cash_flow_dollar_expect = tot_cash_margin_expect * paid_metal_prod_kt_expect * (1e3) \
            - overhead_dollar_annual - sustain_dollar_annual

        # If cash flow expected is negative, determine if mine should incur ramp down this year and reclamation next year, 
        # based on max NPV
        if cash_flow_dollar_expect < 0 and not ramp_up_flag:
            ### Scenario 1: ramp down this year, reclaim the next ###
            ore_prod_kt_ramp_down_this=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, 
                                                           tcm_base, prod_elas, ramp_flag=True)
            og_percent_ramp_down_this=og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.4)**og_elas
            recovered_metal_prod_kt_ramp_down_this = ore_prod_kt_ramp_down_this * og_percent_ramp_down_this * recovery_percent \
            / (1e4)
            paid_metal_prod_kt_ramp_down_this = recovered_metal_prod_kt_ramp_down_this * paid_percent / (1e2)
            cash_flow_dollar_ramp_down_this = tot_cash_margin_expect * paid_metal_prod_kt_ramp_down_this * (1e3) \
            - overhead_dollar_annual - sustain_dollar_annual
            npv_close_this_year = cash_flow_dollar_ramp_down_this - reclamation_dollar/(1+discount)

            ### Scenario 2: continue operation this year, ramp down next year, reclaim two years from now ###
            ore_prod_kt_ramp_down_next=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, 
                                                           tcm_base, prod_elas, ramp_flag=True)
            og_percent_ramp_down_next=og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.75+0.4)**og_elas
            recovered_metal_prod_kt_ramp_down_next = ore_prod_kt_ramp_down_next * og_percent_ramp_down_next * recovery_percent \
            / (1e4)
            paid_metal_prod_kt_ramp_down_next = recovered_metal_prod_kt_ramp_down_next * paid_percent / (1e2)
            cash_flow_dollar_ramp_down_next = tot_cash_margin_expect * paid_metal_prod_kt_ramp_down_next * (1e3) \
            - overhead_dollar_annual - sustain_dollar_annual
            npv_close_next_year = cash_flow_dollar_expect + cash_flow_dollar_ramp_down_next/(1+discount)\
            - reclamation_dollar/(1+discount)**2
            
            if npv_close_this_year > npv_close_next_year:
                # This year at 40% cap uti
                mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt_ramp_down_this
                mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt_ramp_down_this
                mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt_ramp_down_this
                mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar_ramp_down_this

                # Next year incurring reclamation
                t_plus_1=t+relativedelta(years=1)
                mine_life_stats.loc[t_plus_1, 'Ore treated (kt)'] = 0
                mine_life_stats.loc[t_plus_1, 'Paid metal production (kt)'] = 0
                mine_life_stats.loc[t_plus_1, 'Recovered metal production (kt)'] = 0
                mine_life_stats.loc[t_plus_1, 'Pre-tax cash flow ($)'] = -reclamation_dollar
                break

        mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt
        mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt
        mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt
        mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar

        cum_ore_prod_kt += ore_prod_kt
    
    return mine_life_stats


###### Simulate mine life of a particular year, give whole dynamic price_series and TCRC ######
def simulate_mine_life_one_year(year_i, init_year, price_series, tcrc_series, hyper_param, mine_data, mine_life_stats_last,
                                close_price_method='max', close_years_back=5, mine_cu_pct_change = 0):
    mine_life_stats = mine_life_stats_last.copy()
    if 'Reclaim' in mine_life_stats.loc[:, 'Ramp flag'].values:
        return mine_life_stats
    ## Static mine data
    tonne_to_lbs = 2204.62
    ore_cap_kt = mine_data.loc['Ore capacity (kt)']
    og_current_percent = mine_data.loc['Head grade (%)']
    og_init_percent = mine_data.loc['Initial ore grade (%)']
    recovery_percent = mine_data.loc['Recovery rate (%)']
    paid_percent = mine_data.loc['Payable percent (%)']
    og_elas=mine_data.loc['OG elas']
    
    minesite_cost_cpp = mine_data.loc['Minesite cost (cents/lb)']
    freight_cpp = mine_data.loc['Transport and offsite (cents/lb)']
    royalty_cpp = mine_data.loc['Royalty (cents/lb)']
    overhead_dollar_annual = mine_data.loc['Overhead cost ($, annual)']
    sustain_dollar_annual = mine_data.loc['Sustaining capex ($, annual)']
    reclamation_dollar = mine_data.loc['Reclamation ($)']

    # For minesite, the constant cost is based on per tonne of ore rather than metal
    minesite_cost_pto = minesite_cost_cpp*tonne_to_lbs/(1e2)*og_current_percent/(1e2)*recovery_percent/(1e2)*paid_percent/(1e2)
    # Read Hyper-parameters
    cap_uti_base = hyper_param.loc['Capacity utilization constant', 'Value'] * (1-mine_cu_pct_change/100)
    tcm_base = hyper_param.loc['Total cash margin constant ($/tonne)', 'Value']
    prod_elas = hyper_param.loc['Production elasticity', 'Value']
    discount = hyper_param.loc['Discount rate', 'Value']

    ## Dynamic mine data
    t = datetime(year_i, 1, 1)
    t_plus_1=t+relativedelta(years=1)
    price_t = price_series.loc[t]
    cum_ore_prod_kt = mine_life_stats.loc[t, 'Cumulative ore treated (kt)']
    if year_i == init_year and year_i == 2018:
        ramp_flag_name = mine_data.loc['Initial ramp flag']
        tcrc_cpp = mine_data.loc['TCRC (cents/lb)']
        og_percent = og_current_percent

    elif year_i == init_year and year_i > 2018:
        ramp_flag_name = mine_data.loc['Initial ramp flag']
        tcrc_cpp = tcrc_series.loc[t]
        og_percent = og_current_percent

    else:
        tcrc_cpp = tcrc_series.loc[t]
        ramp_flag_name = mine_life_stats.loc[t, 'Ramp flag']
        og_percent = og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.84)**og_elas 
        # 0.84 is the first order approx of last year cap uti    

    ramp_up_flag = False
    ramp_down_flag = False
    if ramp_flag_name == 'Normal':
        pass
    elif ramp_flag_name == 'Up':
        ramp_up_flag = True
    elif ramp_flag_name == 'Down':
        ramp_down_flag = True
    
    ramp_flag=ramp_up_flag or ramp_down_flag

    # Calculate total cash margin and cash flow based on both static and dynamic mine data
    cash_costs = tot_cash_cost_cal(price_t, minesite_cost_pto, og_percent, recovery_percent, paid_percent, 
                                   tcrc_cpp, freight_cpp, royalty_cpp)
    tot_cash_margin = tot_cash_margin_cal(price_t, minesite_cost_pto, og_percent, recovery_percent, paid_percent, 
                                          tcrc_cpp, freight_cpp, royalty_cpp)
    
    ore_prod_kt=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin, tcm_base, prod_elas, ramp_flag)
    recovered_metal_prod_kt = ore_prod_kt * og_percent * recovery_percent / (1e4)
    paid_metal_prod_kt = recovered_metal_prod_kt * paid_percent / (1e2)
    cash_flow_dollar = tot_cash_margin * paid_metal_prod_kt * (1e3) - overhead_dollar_annual - sustain_dollar_annual

    mine_life_stats.loc[t, 'Head grade (%)'] = og_percent
    mine_life_stats.loc[t, 'Total cash margin ($/tonne paid metal)'] = tot_cash_margin
    mine_life_stats.loc[t, ['Minsite cost ($/tonne paid metal)', 'TCRC ($/tonne paid metal)', 
                            'Frieght ($/tonne paid metal)', 'Royalty ($/tonne paid metal)']] = cash_costs.values
    mine_life_stats.loc[t_plus_1, 'Ramp flag'] = 'Normal'

    # If closing already determined at this point (ramp_down_flag=True), execute closing produciton
    if ramp_down_flag:
        # This year at 40% cap uti
        mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt
        mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt
        mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt
        mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar
        mine_life_stats.loc[t, 'Ramp flag'] = 'Down'
        
        # Next year incurring reclamation
        mine_life_stats.loc[t_plus_1, 'Ore treated (kt)'] = 0
        mine_life_stats.loc[t_plus_1, 'Paid metal production (kt)'] = 0
        mine_life_stats.loc[t_plus_1, 'Recovered metal production (kt)'] = 0
        mine_life_stats.loc[t_plus_1, 'Pre-tax cash flow ($)'] = -reclamation_dollar
        mine_life_stats.loc[t_plus_1, 'Ramp flag'] = 'Reclaim'
        return mine_life_stats
        
    # If not closing not determined yet, calculate cashflow under expected price, not assuming ramp down yet. If this cash
    # flow turns out to be negative, begin to calculate the ramp down cash flow
    else:
        # Expected price: assume miners look back on historical price as well, use trailing price (3 years trailing for now)
        if close_price_method=='mean':
            price_expect = price_series.rolling(close_years_back).mean().loc[t]
        elif close_price_method=='max':
            t_back=t-relativedelta(years=close_years_back-1)
            price_expect = price_series.loc[t_back:t].max()

        tot_cash_margin_expect = tot_cash_margin_cal(price_expect, minesite_cost_pto, og_percent, recovery_percent, 
                                                     paid_percent, tcrc_cpp, freight_cpp, royalty_cpp)            
        ore_prod_kt_expect=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, tcm_base, prod_elas, 
                                               ramp_flag=False)
        recovered_metal_prod_kt_expect = ore_prod_kt_expect * og_percent * recovery_percent / (1e4)
        paid_metal_prod_kt_expect = recovered_metal_prod_kt_expect * paid_percent / (1e2)
        cash_flow_dollar_expect = tot_cash_margin_expect * paid_metal_prod_kt_expect * (1e3) \
        - overhead_dollar_annual - sustain_dollar_annual

    # If cash flow expected is negative, determine if mine should incur ramp down this year and reclamation next year, 
    # based on max NPV
    if cash_flow_dollar_expect < 0 and not ramp_up_flag:
        ### Scenario 1: ramp down this year, reclaim the next ###
        ore_prod_kt_ramp_down_this=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, 
                                                       tcm_base, prod_elas, ramp_flag=True)
        og_percent_ramp_down_this=og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.4)**og_elas
        recovered_metal_prod_kt_ramp_down_this = ore_prod_kt_ramp_down_this * og_percent_ramp_down_this * recovery_percent \
        / (1e4)
        paid_metal_prod_kt_ramp_down_this = recovered_metal_prod_kt_ramp_down_this * paid_percent / (1e2)
        cash_flow_dollar_ramp_down_this = tot_cash_margin_expect * paid_metal_prod_kt_ramp_down_this * (1e3) \
        - overhead_dollar_annual - sustain_dollar_annual
        npv_close_this_year = cash_flow_dollar_ramp_down_this - reclamation_dollar/(1+discount)

        ### Scenario 2: continue operation this year, ramp down next year, reclaim two years from now ###
        ore_prod_kt_ramp_down_next=ore_prod_calculator(ore_cap_kt, cap_uti_base, tot_cash_margin_expect, 
                                                       tcm_base, prod_elas, ramp_flag=True)
        og_percent_ramp_down_next=og_init_percent * (cum_ore_prod_kt/ore_cap_kt+0.75+0.4)**og_elas
        recovered_metal_prod_kt_ramp_down_next = ore_prod_kt_ramp_down_next * og_percent_ramp_down_next * recovery_percent \
        / (1e4)
        paid_metal_prod_kt_ramp_down_next = recovered_metal_prod_kt_ramp_down_next * paid_percent / (1e2)
        cash_flow_dollar_ramp_down_next = tot_cash_margin_expect * paid_metal_prod_kt_ramp_down_next * (1e3) \
        - overhead_dollar_annual - sustain_dollar_annual
        npv_close_next_year = cash_flow_dollar_expect + cash_flow_dollar_ramp_down_next/(1+discount)\
        - reclamation_dollar/(1+discount)**2

        if npv_close_this_year > npv_close_next_year:
            # This year at 40% cap uti
            mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt_ramp_down_this
            mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt_ramp_down_this
            mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt_ramp_down_this
            mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar_ramp_down_this
            mine_life_stats.loc[t, 'Ramp flag'] = 'Down'
            
            # Next year incurring reclamation
            mine_life_stats.loc[t_plus_1, 'Ore treated (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Paid metal production (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Recovered metal production (kt)'] = 0
            mine_life_stats.loc[t_plus_1, 'Pre-tax cash flow ($)'] = -reclamation_dollar
            mine_life_stats.loc[t_plus_1, 'Ramp flag'] = 'Reclaim'
            
            return mine_life_stats
            
    if ramp_up_flag or mine_life_stats.loc[t, 'Ramp flag'] == 'Normal':
        mine_life_stats.loc[t, 'Ore treated (kt)'] = ore_prod_kt
        mine_life_stats.loc[t, 'Paid metal production (kt)'] = paid_metal_prod_kt
        mine_life_stats.loc[t, 'Recovered metal production (kt)'] = recovered_metal_prod_kt
        mine_life_stats.loc[t, 'Pre-tax cash flow ($)'] = cash_flow_dollar

    mine_life_stats.loc[t_plus_1, 'Cumulative ore treated (kt)']=cum_ore_prod_kt+ore_prod_kt

    return mine_life_stats

def mine_life_init(simulation_time, mine_data, init_year):
    init_time=datetime(init_year, 1, 1)
    mine_life_stats_init = pd.DataFrame(index=simulation_time, 
                                        columns=['Head grade (%)', 'Ore treated (kt)', 'Cumulative ore treated (kt)',
                                                 'Recovered metal production (kt)', 'Paid metal production (kt)', 
                                                 'Total cash margin ($/tonne paid metal)', 'Minsite cost ($/tonne paid metal)', 
                                                 'TCRC ($/tonne paid metal)', 'Frieght ($/tonne paid metal)', 
                                                 'Royalty ($/tonne paid metal)', 'Pre-tax cash flow ($)', 'Ramp flag'])
    mine_life_stats_init.loc[init_time, 'Cumulative ore treated (kt)'] = mine_data.loc['Cumulative ore treated (kt)']
    mine_life_stats_init.loc[init_time, 'Ramp flag'] = mine_data.loc['Initial ramp flag']
    return mine_life_stats_init


def mine_life_stats_panel_init(simulation_time, mine_pool):
    if mine_pool.shape[0] == 0:
        return pd.DataFrame()
    else:
        mine_properties=mine_life_init(simulation_time, mine_pool.iloc[0, :], init_year=2018).columns
        iterables = [mine_pool.index, mine_properties]
        columns_mi=pd.MultiIndex.from_product(iterables, names=['Mine ID', 'Values'])
        mine_life_stats_panel=pd.DataFrame(0, index=simulation_time, columns=columns_mi)
        return mine_life_stats_panel


def simulate_mine_life_stats_panel(simulation_time, year_i, init_year, 
                                   mine_life_stats_panel, mine_data, 
                                   price_series, tcrc_series, hyper_param,
                                   close_price_method='max', close_years_back=5, mine_cu_pct_change = 0):
    if year_i == init_year:
        mine_life_stats_init = mine_life_init(simulation_time, mine_data, init_year)
        mine_life_stats = simulate_mine_life_one_year(year_i, init_year, price_series, tcrc_series, hyper_param, 
                                                      mine_data, mine_life_stats_init, 
                                                      close_price_method, close_years_back)

    else: 
        mine_life_stats_last_temp = mine_life_stats_panel.loc[:, idx[mine_data.name, :]]
        mine_life_stats_last = mine_life_stats_last_temp.copy()
        mine_life_stats_last.columns = mine_life_stats_last_temp.columns.droplevel()
        mine_life_stats = simulate_mine_life_one_year(year_i, init_year, price_series, tcrc_series, hyper_param, 
                                                      mine_data, mine_life_stats_last, 
                                                      close_price_method, close_years_back, 
                                                      mine_cu_pct_change=mine_cu_pct_change)
    
    return mine_life_stats


def og_elas_calculator_reserve(mine_data, price_series, hyper_param, 
                               reserve_valid, og_elas_init=0, verbose=0, ratio_diff_cutoff=0.05):
    # Initial
    og_elas=og_elas_init
    step=0.1
    ratio_diff=1
    direction=-1
    iteration=0
    simulation_end_time='21000101'
    simulation_time=pd.date_range('20180101', simulation_end_time, freq='AS')
    
    # If no ore grade decline and still cannot deplete reserve before 2100, then return 0.
    mine_life_stat_no_decline=\
    simulate_mine_life(simulation_time=simulation_time, 
                       price_series=price_series, 
                       hyper_param=hyper_param, 
                       mine_data=mine_data, og_elas=0)
    simulated_reserve_no_decline=mine_life_stat_no_decline.loc[:, 'Recovered metal production (kt)'].sum()

    if simulated_reserve_no_decline < reserve_valid:
        return 0
    
    while ratio_diff > ratio_diff_cutoff:
        iteration=iteration+1
        if iteration>50:
            return 'No convergence'
        
        last_direction=direction
        og_elas=og_elas+step*direction

        mine_life_stat=\
        simulate_mine_life(simulation_time=simulation_time, 
                           price_series=price_series, 
                           hyper_param=hyper_param, 
                           mine_data=mine_data, og_elas=og_elas)
        simulated_reserve=mine_life_stat.loc[mine_life_stat.loc[:, 'Pre-tax cash flow ($)']>0, 
                                             'Recovered metal production (kt)'].sum()
        # Those years under negative cash flow are not considered to be producing reserves
        
        ratio_diff=abs(simulated_reserve-reserve_valid)/reserve_valid

        if simulated_reserve < reserve_valid:
            direction=1
        elif simulated_reserve > reserve_valid:
            direction=-1

        if last_direction+direction==0:
            step=step*0.1
            
        if verbose == 1:
            print(og_elas, last_direction, direction)
        
    return og_elas


def og_elas_calculator_year(mine_data, price_series, hyper_param, 
                            last_year_benchmark, og_elas_init=0, verbose=0):
    # Initial
    og_elas=og_elas_init
    step=0.1
    direction=-1
    last_year=2100
    simulation_end_time='21000101'
    simulation_time=pd.date_range('20180101', simulation_end_time, freq='AS')

    while last_year != last_year_benchmark:
        last_direction=direction
        og_elas=og_elas+step*direction

        mine_life_stat=\
        simulate_mine_life(simulation_time=simulation_time, 
                           price_series=price_series, 
                           hyper_param=hyper_param, 
                           mine_data=mine_data, og_elas=og_elas)
        last_year=(mine_life_stat.loc[mine_life_stat.loc[:, 'Pre-tax cash flow ($)']>0, 
                                      'Recovered metal production (kt)'].index[-1]).year
 
        if last_year < last_year_benchmark:
            direction=1
        elif last_year > last_year_benchmark:
            direction=-1

        if last_direction+direction==0:
            step=step*0.1
            
        if verbose == 1:
            print(og_elas, last_direction, direction)
        
    return og_elas


def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)


def irr(cfs, yrs, x0, **kwargs):
    return np.asscalar(fsolve(npv, x0=x0, args=(cfs, yrs), **kwargs))


def irr_filter(simulation_time, price_series, tcrc_series, hyper_param, mine_data, irr_cutoff, 
               close_price_method='max', close_years_back=5):
    mine_life_stat=simulate_mine_life(simulation_time, price_series, tcrc_series, hyper_param, 
                                      mine_data, og_elas=mine_data.loc['OG elas'], 
                                      close_price_method=close_price_method,
                                      close_years_back=close_years_back)
    initial_capex=mine_data.loc['Developement capex ($, annual)']
    cash_flow=np.append(np.repeat(-initial_capex, 3), mine_life_stat.loc[:, 'Pre-tax cash flow ($)'].fillna(0).values)
    
    npv_estimate=npv(irr_cutoff, cash_flow, np.arange(cash_flow.size))
    if npv_estimate > 0:
        return True
    else:
        return False    
 


def total_production_calculator_multi_index(simulation_time, hyper_param, price_series, tcrc_series, operating_mine_data, 
                                            close_price_method='max', close_years_back=5):
    iterables = [operating_mine_data.index, ['Ore treated (kt)', 'Recovered metal production (kt)', 'Head grade (%)']]
    columns_mi=pd.MultiIndex.from_product(iterables, names=['Mine ID', 'Values'])
    total_production=pd.DataFrame(0, index=simulation_time, columns=columns_mi)
    
    for i in operating_mine_data.index:
        mine_data=operating_mine_data.loc[i, :]
        og_elas=operating_mine_data.loc[i, 'OG elas']
        mine_life_stat=simulate_mine_life(simulation_time, price_series, tcrc_series, hyper_param, mine_data, og_elas, 
                                          close_price_method, close_years_back)
        mine_life_stat=mine_life_stat.fillna(0)
        total_production.loc[:, idx[i, :]]=mine_life_stat.loc[simulation_time, ['Ore treated (kt)', 
                                                                  'Recovered metal production (kt)', 'Head grade (%)']].values

    return total_production


def new_mine_data(year, simulation_end_time, incentive_pool, price_series, tcrc_series, hyper_param, 
                  subsample_size, irr_cutoff, close_price_method='max', close_years_back=5,
                  trailing_years=10, block_size=2000):
    # year is mine opening year, not year when opening decision is made
    simulation_time=pd.date_range(datetime(year,1,1), simulation_end_time, freq='AS')
    # Use -3 year trailing price as projection for the future
    price_series_trailing_av=price_series.rolling(trailing_years).mean().loc[datetime(year-3, 1, 1)]
    price_expected=pd.Series(price_series_trailing_av, index=simulation_time)
    
    tcrc_series_trailing_av=tcrc_series.rolling(trailing_years).mean().loc[datetime(year-3, 1, 1)]
    tcrc_expected=pd.Series(tcrc_series_trailing_av, index=simulation_time)
    # Skipping block_size every year
    incentive_id_subsample=incentive_pool.index[(year-2019)*block_size:(year-2019)*block_size+subsample_size]
    irr_filtered_subsample=pd.Series(False, index=incentive_id_subsample)

    for i in incentive_id_subsample:
        open_indicator=irr_filter(simulation_time, price_expected, tcrc_expected, hyper_param, 
                                  mine_data=incentive_pool.loc[i, :],
                                  irr_cutoff=irr_cutoff, 
                                  close_price_method=close_price_method,
                                  close_years_back=close_years_back)
        irr_filtered_subsample.loc[i]=open_indicator

    new_mine=incentive_pool.loc[incentive_id_subsample, :].loc[irr_filtered_subsample, :]
    new_mine['Initial year']=year
    return new_mine


def total_production_calculator_newmine_multi_index(open_parameter, simulation_end_time, incentive_pool, 
                                                    price_series, tcrc_series, hyper_param, 
                                                    close_price_method='max', close_years_back=5):
    iterables = [['temp'], ['Ore treated (kt)', 'Recovered metal production (kt)', 'Head grade (%)']]
    columns_mi=pd.MultiIndex.from_product(iterables, names=['Mine ID', 'Values'])
    tot_prod_mi=pd.DataFrame(0, index=pd.date_range('20180101', simulation_end_time, freq='AS'), columns=columns_mi)

    for year in open_parameter.index:
        print(year)
        subsample_size=open_parameter.loc[year, 'Subsample size']
        mine_data_new_year=new_mine_data(year, simulation_end_time, incentive_pool, price_series, tcrc_series,
                                         hyper_param, subsample_size, 0.15,
                                         close_price_method, close_years_back)
        print('Found mines that could open: ', mine_data_new_year.shape[0])
        if mine_data_new_year.shape[0]>0:
            tot_prod_mi_this_year=total_production_calculator_multi_index(pd.date_range(datetime(year,1,1), simulation_end_time, freq='AS'),
                                                                         hyper_param, price_series, tcrc_series, mine_data_new_year,
                                                                         close_price_method, close_years_back)
            tot_prod_mi=pd.concat([tot_prod_mi, tot_prod_mi_this_year], axis=1)
        print('New production calculated')

    return tot_prod_mi


def total_production_calculator_newmine_multi_index_cali(open_parameter, simulation_end_time, incentive_pool, 
                                                         price_series, tcrc_series, hyper_param, 
                                                         close_price_method, close_years_back, cali_start, cali_end):
    iterables = [['temp'], ['Ore treated (kt)', 'Recovered metal production (kt)', 'Head grade (%)']]
    columns_mi=pd.MultiIndex.from_product(iterables, names=['Mine ID', 'Values'])
    tot_prod_mi=pd.DataFrame(0, index=pd.date_range('20180101', simulation_end_time, freq='AS'), columns=columns_mi)

    for year in np.arange(cali_start, cali_end+1):
        print(year)
        subsample_size=open_parameter.loc[year, 'Subsample size']
        mine_data_new_year=new_mine_data(year, simulation_end_time, incentive_pool, price_series, tcrc_series,
                                         hyper_param, subsample_size, 0.15,
                                         close_price_method, close_years_back)
        print('Found mines that could open: ', mine_data_new_year.shape[0])
        if mine_data_new_year.shape[0]>0:
            tot_prod_mi_this_year=total_production_calculator_multi_index(pd.date_range(datetime(year,1,1), simulation_end_time, freq='AS'),
                                                                         hyper_param, price_series, tcrc_series, mine_data_new_year,
                                                                         close_price_method, close_years_back)
            tot_prod_mi=pd.concat([tot_prod_mi, tot_prod_mi_this_year], axis=1)
        print('New production calculated')

    return tot_prod_mi



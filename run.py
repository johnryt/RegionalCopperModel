import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
idx = pd.IndexSlice
import random
from gurobipy import *
from datetime import datetime

from cn_mine_simulation_tools import *
from cn_refinery import *
from cn_scrap_supply_tools import *
from cn_demand_tools import *
from cn_price_formation import * 
from cn_blending import *

import warnings
warnings.filterwarnings("ignore") # to deal with pandas datetime deprecation

if __name__=='__main__':
    ## High level parameters
    historical_prod=pd.read_excel('Data/Production data compile.xlsx', sheet_name='Selected', index_col=0).loc[:2018]
    historical_prod_cn=pd.read_excel('Data/Production data compile.xlsx', sheet_name='China', index_col=0, usecols='A:O,W:X,AE').loc[:2018]
    historical_prod_rw = historical_prod.loc[1950:] - historical_prod_cn.loc[1950:]
    historical_price=pd.read_excel('Data/Price data compile.xlsx', sheet_name='Price', index_col=0) # All prices 2017 constant
    historical_price.drop(['Grape','Low_brass','Pb_Red_Brass'], axis = 1, inplace = True)
    ref_prices_future = pd.read_csv('Data/Future metal prices.csv',index_col=0).loc[:,'Ref_Zn':]
    ref_prices_historical = historical_price.loc[:,'Ref_Zn':'Ref_Fe'].copy()
    ref_prices_historical.drop(2018)
    ref_prices_future.index = list(range(2018,2048))
    ref_prices = pd.concat([ref_prices_historical,ref_prices_future])

    # Specific prod and mine data
    historical_mining_prod=historical_prod.loc[:, 'Total mining production'].copy()
    historical_mining_prod_cn = historical_prod_cn.loc[:, 'Total mining production'].copy()
    historical_mining_prod_rw = historical_prod_rw.loc[:, 'Total mining production'].copy()

    historical_lme_price=historical_price.loc[:, 'LME'].copy()
    ref_price=pd.concat([historical_price.loc[:,'Ref_Zn':'Ref_Fe'],ref_prices_future])
    historical_tcrc=historical_price.loc[:, 'Annual TCRC'].copy()
    historical_no2=historical_price.loc[:, 'No.2 ref'].copy()
    historical_sp2=historical_lme_price-historical_no2
    historical_no1 = historical_price.loc[:,'Barley brass']
    historical_sp1 = historical_lme_price-historical_no1
    historical_alloyed = historical_price.loc[:,'Yellow_Brass':'Cartridge']
    historical_alloyed = historical_alloyed[sorted(list(historical_alloyed.columns))]
    scraps = list(historical_alloyed.columns)
    historical_spa = historical_alloyed.apply(lambda x: -x + historical_lme_price)
    raw_price = pd.concat([historical_price.loc[:,'Ref_Cu':'Cartridge'],ref_prices_future.loc[2019:,:]],sort=False)
    raw_price.loc[:,'No.1'] = historical_no1
    raw_price.loc[:,'No.2'] = historical_no2
    raw_price.loc[2019:,:] = 0
    ref_metals = list(ref_price.columns)
    for i in ['Cartridge','Pb_Yellow_Brass']:
        raw_price.loc[1999:2018,i] = raw_price.loc[1999:2018,'Yellow_Brass'] * raw_price.loc[2018,i] / raw_price.loc[2018,'Yellow_Brass']
    for i in ['Al_Bronze','Sn_Bronze','Pb_Sn_Bronze']:
        raw_price.loc[1999:2018,i] = raw_price.loc[1999:2018,'Red_Brass'] * raw_price.loc[2018,i] / raw_price.loc[2018,'Red_Brass']
    for i in ['Mn_Bronze','Ni_Ag']:
        raw_price.loc[1999:2018,i] = raw_price.loc[1999:2018,'Ocean'] * raw_price.loc[2018,i] / raw_price.loc[2018,'Ocean']
        # used their compositions to decide which of our real-data prices to follow - subjective

    ## Primary supply data and patameters
    operating_mine_pool=pd.read_excel('Data/primary supply/Operating mine pool - countries.xlsx', sheet_name='Sheet1', index_col=0)
    open_parameter=pd.read_excel('Data/primary supply/Opening subsample parameter.xlsx', sheet_name='max5', index_col=0)
    incentive_pool=pd.read_excel('Data/primary supply/Incentive mine pool - countries.xlsx', sheet_name='Sheet1', index_col=0)
    pri_hyper_param=pd.read_excel('Data/primary supply/Hyperparameters.xlsx', sheet_name='Sheet1', index_col=0)
    operating_mine_pool_cn = operating_mine_pool.loc[operating_mine_pool.loc[:, 'Country'] == 'China']
    operating_mine_pool_rw = operating_mine_pool.loc[operating_mine_pool.loc[:, 'Country'] != 'China']
    incentive_pool_cn = incentive_pool.loc[incentive_pool.loc[:, 'Country'] == 'China']
    incentive_pool_rw = incentive_pool.loc[incentive_pool.loc[:, 'Country'] != 'China']

    ## Refinery parameters
    ref_hyper_param=pd.read_excel('Data/refined supply/Refinery hyperparameter.xlsx', sheet_name='Parameters', index_col=0)
    ref_hyper_param_cn = pd.read_excel('Data/refined supply/Refinery hyperparameter.xlsx', sheet_name='CN Parameters', index_col=0)
    ref_hyper_param_rw = pd.read_excel('Data/refined supply/Refinery hyperparameter.xlsx', sheet_name='RW Parameters', index_col=0)
    conc_to_cathode_eff=ref_hyper_param.loc['conc to cathode eff', 'Value']
    scrap_to_cathode_eff=ref_hyper_param.loc['scrap to cathode eff', 'Value']
    ref_prod_history = historical_prod.loc[1960:, 'Primary refining production':'Refined usage']
    ref_prod_history_cn = historical_prod_cn.loc[1960:, 'Primary refining production':'Refined production, WoodMac'].copy()
    ref_prod_history_cn.columns = ref_prod_history.columns
    ref_prod_history_rw = ref_prod_history - ref_prod_history_cn
    historical_ref_imports_cn = pd.Series(0,index=np.arange(1960,2041))
    historical_ref_imports_cn.loc[:2018] = historical_prod_cn.loc[:, 'Net Refined Imports'].copy()
    historical_ref_imports_cn.loc[2019:] = historical_ref_imports_cn.loc[2018]
    historical_ref_imports_cn.loc[:1974] += historical_prod_cn.loc[:1974,'Semis Imports (COMTRADE, kt)']
    og_historical_ref_imports_cn = historical_ref_imports_cn.copy()
    # ref_prod_history_cn.loc[:, 'Refined usage'] -= historical_ref_imports_cn.loc[1960:]
    # ref_prod_history_rw.loc[:, 'Refined usage'] += historical_ref_imports_cn.loc[1960:]

    ## Semis demand parameters
    gdp_growth_prediction_base=pd.read_excel('Data/semis demand/Demand prediction data.xlsx', sheet_name='GDP growth', index_col=0, usecols=np.arange(6))
    volume_prediction_base=pd.read_excel('Data/semis demand/Demand prediction data.xlsx', sheet_name='All sectors', index_col=0, header=[0,1])
    intensity_prediction=pd.read_excel('Data/semis demand/Intensity initial.xls', sheet_name='Sheet1', index_col=0, header=[0,1])
    elas_sec_reg=pd.read_excel('Data/semis demand/Elasticity estimates.xlsx', sheet_name='S+R S intercept only', index_col=0)
    sector_shape_matrix=pd.read_excel('Data/semis demand/Sector to shape matrix updated.xlsx', sheet_name='Sheet1', index_col=0)
    calibration_1718=pd.read_excel('Data/semis demand/2017 and 2018 calibration.xlsx', sheet_name='Sheet1', index_col=0)
    wire_mill_fraction = 0.736162363 # fraction of unalloyed production coming from wire mills and not copper brass mills, since wire mills cannot use No.1 but brass mills can
    copper_brass_mill_fraction = 1 - wire_mill_fraction
    fruity_alloys = pd.read_excel('Data/semis demand/Fruity alloys.xlsx', sheet_name='Sheet1', index_col=0)
    og_fruity_alloys = fruity_alloys.copy()
    fruity_alloyed = fruity_alloys.copy().loc[fruity_alloys.loc[:,'Alloy Type'] != 'No.1']
    fruity_alloyed.loc['Fruity No.1',:] = fruity_alloys.loc['Mainboard',:].copy()
    fruity_alloyed.loc['Fruity No.1','Quantity'] = fruity_alloys.loc[fruity_alloys.loc[:,'Alloy Type']=='No.1','Quantity'].sum()
    og_fruity_alloyed = fruity_alloyed.copy()

    # Adjust demand in 2018 to scale it back to ICSG
    intensity_prediction.loc[2017, :] = intensity_prediction.loc[2017, :]\
    .mul(calibration_1718.loc[2017, 'ICSG refined usage']).div(calibration_1718.loc[2017, 'simulated refined usage'])
    intensity_prediction.loc[2018, :] = intensity_prediction.loc[2018, :]\
    .mul(calibration_1718.loc[2018, 'ICSG refined usage']).div(calibration_1718.loc[2018, 'simulated refined usage'])
    demand_prediction=volume_prediction_base.loc[2015:, :].mul(intensity_prediction.fillna(0))

    ## Scrap supply parameters
    use_sector_combined=pd.read_excel('Data/scrap supply/End use combined data.xlsx', sheet_name='Combined', index_col=0)
    sector_to_product=pd.read_excel('Data/scrap supply/All accounting matrix.xlsx', sheet_name='sector to product', index_col=0)
    product_to_waste=pd.read_excel('Data/scrap supply/All accounting matrix.xlsx', sheet_name='product to waste', index_col=0)
    product_life_and_eff=pd.read_excel('Data/scrap supply/All accounting matrix.xlsx', sheet_name='product lifetime and efficiency', index_col=0)
    product_to_cathode_alloy=pd.read_excel('Data/scrap supply/All accounting matrix.xlsx', sheet_name='product to copper or alloy', index_col=0)
    recycle_efficiency=pd.read_excel('Data/scrap supply/All accounting matrix.xlsx', sheet_name='recycling efficiency', index_col=0)
    fraction_no1_old = pd.Series({'Plumbing': 0.7,'Building Plant': 0.7,'Architecture': 0.7, 'Communications': 0.5, 
                            'Electrical Power': 0.7,'Telecommunications': 0.7,'Power Utility': 0.7,
                            'Electrical Industrial': 0.2, 'Non Elec. Industrial':0.1, 'Electrical Automotive': 0.1,
                            'Non Elec. Automotive': 0.1, 'Other Transport': 0.1, 'Consumer': 0.1, 'Cooling': 0.1, 
                            'Electronic': 0.1, 'Diverse': 0.1})
    fraction_no1_new = pd.Series({'Plumbing': 0.9,'Building Plant': 0.9,'Architecture': 0.9, 'Communications': 0.9, 
                            'Electrical Power': 0.9,'Telecommunications': 0.9,'Power Utility': 0.9,
                            'Electrical Industrial': 0.7, 'Non Elec. Industrial':0.7, 'Electrical Automotive': 0.5,
                            'Non Elec. Automotive': 0.7, 'Other Transport': 0.7, 'Consumer': 0.7, 'Cooling': 0.8, 
                            'Electronic': 0.3, 'Diverse': 0.7})

    # Availability-specific parameters
    s2s = pd.read_excel('Data/Shape-Sector Distributions.xlsx', index_col=0)
    prod_spec = pd.read_excel('Data/Prod_spec_20200311_no_low.xlsx')
    raw_spec = pd.read_excel('Data/Raw_spec_201901.xlsx',index_col=0)
    raw_spec.drop(['Grape','Low_brass','Pb_Red_Brass'],inplace=True)
    for i in prod_spec.index:
        prod_spec.loc[i,'UNS'] = prod_spec.loc[i,'UNS']+' '+prod_spec.loc[i,'Category']

    # raw_spec.loc['Pb_Sn_Bronze':'Pb_Yellow_Brass','High_Ni':'High_Mn'] = 0
        
    # Home scrap ratio
    home_scrap_ratio_file=pd.read_excel('Data/scrap supply/Home scrap ratio.xls', sheet_name='Sheet1', index_col=0)
    home_scrap_ratio_series=home_scrap_ratio_file.loc[:, 'Calibrated ratio']
    exchange_scrap_ratio_series=0.9-home_scrap_ratio_series

    # Sector end use to product matrix 
    use_product_history = use_sector_combined.apply(lambda x: (x*sector_to_product).sum(axis=1),axis=1)
    # use_product_history=pd.DataFrame(np.matmul(use_sector_combined, sector_to_product.transpose()), 
    #                                  index=use_sector_combined.index, columns=sector_to_product.index)
    demand_fractions = pd.read_excel('Data/semis demand/demand_analysis_copper_lto_q2_2016.xls', sheet_name='Analysis', index_col = 0)
    cn_demand_fraction = demand_fractions.loc[:,'China Fraction']
    rw_demand_fraction = 1 - cn_demand_fraction
    use_product_history_cn = use_product_history.apply(lambda x: x*cn_demand_fraction.loc[:2018])
    use_product_history_rw = use_product_history.apply(lambda x: x*rw_demand_fraction.loc[:2018])

    # Product to waste matrices
    product_to_waste_collectable=product_to_waste.iloc[:, :-2]
    product_to_waste_no_loss=product_to_waste_collectable.mul(1/product_to_waste_collectable.sum(axis=1), axis=0)

    # Product lifetime parameters and frequencies
    product_lifetime=product_life_and_eff.loc[:, 'Lifetime']
    product_lifetime_cn = product_life_and_eff.loc[:, 'CN Lifetime']
    product_lifetime_df=lifetime_df(product_lifetime)
    product_lifetime_df_cn = lifetime_df(product_lifetime_cn)
    product_lifetime_freq_df=lifetime_freq_df(product_lifetime_df)
    product_lifetime_freq_df_cn = lifetime_freq_df(product_lifetime_df_cn)

    # Recycling and fabrication efficiencies
    sort_eff=recycle_efficiency.iloc[:, 0]
    sort_eff_cn = recycle_efficiency.iloc[:, 2]
    collect_rate=recycle_efficiency.iloc[:, 1]
    collect_rate_cn = recycle_efficiency.iloc[:, 3]
    fab_eff=product_life_and_eff.loc[:, 'Fabrication efficiency']
    fab_eff_cn = product_life_and_eff.loc[:, 'CN Fabrication efficiency']
    new_scrap_gen=1/fab_eff-1
    new_scrap_gen_cn = 1/fab_eff_cn-1
    sort_eff_series=pd.DataFrame(np.array((list(sort_eff)*23)).reshape(23, 6), index=np.arange(2018, 2041), columns=sort_eff.index)
    sort_eff_series_cn = pd.DataFrame(np.array((list(sort_eff_cn)*23)).reshape(23, 6), index = np.arange(2018,2041), columns = sort_eff_cn.index)
    collect_rate_series=pd.DataFrame(np.array((list(collect_rate)*23)).reshape(23, 6), index=np.arange(2018, 2041), columns=collect_rate.index)
    collect_rate_series_cn=pd.DataFrame(np.array((list(collect_rate_cn)*23)).reshape(23, 6), index=np.arange(2018, 2041), columns=collect_rate_cn.index)

    ## Price formation parameters
    price_formation_param=pd.read_excel('Data/price formation/Price formation.xlsx', sheet_name='Sheet1', index_col=0)
    cathode_sd_elas=price_formation_param.loc['Cathode SD elasticity', 'Value']
    conc_sd_elas=price_formation_param.loc['Concentrate SD elasticity', 'Value']
    cathode_sp2_elas=price_formation_param.loc['SP2 cathode elasticity', 'Value']
    sp2_sd_elas=price_formation_param.loc['SP2 SD elasticity', 'Value']
    cathode_sp1_elas=price_formation_param.loc['SP1 cathode elasticity', 'Value']
    sp1_sd_elas=price_formation_param.loc['SP1 SD elasticity', 'Value']
    cathode_alloyed_elas=price_formation_param.loc['SP2 SD elasticity', 'Value']
    alloyed_sd_elas=price_formation_param.loc['SP Alloy SD elasticity', 'Value']

    print('Importing data and parameters complete: '+str(datetime.now()))

######### SYSTEM INITIALIZATION ##############
        # Set rollover to 1 for previous year's scrap becoming the new year's scrap
    rollover = 1
    # Set scrappy to 1 for scrap balance being determined by scrap consumption from blending rather than semis demand, set it to 2 to only use the subset of scraps described by scrap_subset variable
    scrappy = 2
    # scrap_subset = ['No.1', 'No.2', 'Cartridge', 'Ocean', 'Red_Brass', 'Yellow_Brass',    'Pb_Yellow_Brass']
    scrap_subset = list(['Yellow_Brass','Cartridge', 'No.1', 'No.2', 'Pb_Yellow_Brass', 'Ocean', 
                        'Red_Brass', 'Al_Bronze', 'Ni_Ag'])
    # scrap_subset = ['No.1']
    # scrap_subset = ['No.1', 'No.2', 'Al_Bronze', 'Cartridge', 'Ni_Ag', 'Ocean',
    #        'Pb_Sn_Bronze', 'Pb_Yellow_Brass', 'Red_Brass', 'Yellow_Brass'] # leaving out tin bronze since it has the lowest use fraction
    # Set include_unalloyed to 1 to expand semis to include unalloyed production, rather than only alloyed
    include_unalloyed = 1
    # Set inventory to 1 to use scrap entering inventory minus scrap leaving inventory as the scrap supply-demand balance rather than all scrap available minus scrap demand
    inventory = 1
    # Set use_Ref_Cu to 1 to use the refined copper demand coming from blending as our copper demand
    use_Ref_Cu = 1
    # Set slow_change != 0 to keep blending from changing so quickly year-over-year, value is percent change permitted per year
    slow_change = 0
    # fraction_yellows is fraction of Yellow_Brass, Pb_Yellow_Brass, and Cartridge allowed in secondary refineries , while unalloyed tune changes the availability of No.1 and No.2 scraps in the blending module. Default 1 means they both have availability equal to total unalloyed quantity
    fraction_yellows = 0.05 # set to zero to avoid this method, which includes secondary refineries in blending
    unalloyed_tune = 1
    use_new_recovery = 0 # set to 1 to use fraction_no1_old and new to determine No.1/No.2 ratio of unalloyed scrap, 0 uses No.2 quantity = secondary refinery demand 
    refined_import_rate = 1 # refined imports, 1 is default
    CU_ref_bal_elas = 0 # random estimate for how much refinery supply-demand ratio impacts CU, should be negative, 0 for default
    fruity_alloys = og_fruity_alloys.copy()
    fruity_alloyed = fruity_alloys.copy().loc[fruity_alloys.loc[:,'Alloy Type'] != 'No.1']

    # fruity_alloyed.loc['Fruity No.1',:] = fruity_alloys.loc['Mainboard',:].copy()
    # fruity_alloyed.loc['Fruity No.1','Quantity'] = fruity_alloys.loc[fruity_alloys.loc[:,'Alloy Type']=='No.1','Quantity'].sum()
    # fruity_alloys.loc[:'Mesa','Quantity'] *= 10
    fruity_multiplier = 1 # 0.1*30997.694215369822/og_fruity_alloys.loc[:,'Quantity'].sum() # 1 is default, 30997 is 2018 direct_melt_sectorial_demand value, the leading value (eg 0.01) would be the fraction of market occupied by these alloys
    pir_pcr = 1
    pir_fraction = -1
    fruity_rr = [0.01,0.01,0.01,0.01]
    scrap_bal_correction = 0.9 # 0.928
    pir_price_set = 0.65

    # Initialize simulation time
    history_start_time='19600101'
    simulation_start_time='20180101'
    simulation_end_time='20400101'
    simulation_time=pd.date_range(simulation_start_time, simulation_end_time, freq='AS')
    history_time=pd.date_range(history_start_time, simulation_start_time, freq='AS')

    # Cathode price
    cathode_price_series=pd.Series(0, index=history_time)
    cathode_price_series.loc[:'20180101']=historical_lme_price.values
    cathode_bal_l1 = pd.Series(0, index = np.arange(2018,2041))

    # TCRC
    tcrc_series=pd.Series(0, index=history_time)
    tcrc_series.loc[:'20180101']=historical_tcrc.values

    # Scrap spreads (No.2, No.1, alloyed)
    sp2_series=pd.Series(0, index=history_time)
    sp2_series.loc[:'20180101']=historical_sp2.values
    sp2_series_cn=pd.Series(0, index=history_time)
    sp2_series_cn.loc[:'20180101']=historical_sp2.values
    sp2_series_rw=pd.Series(0, index=history_time)
    sp2_series_rw.loc[:'20180101']=historical_sp2.values
    sp1_series = pd.Series(0, index=history_time)
    sp1_series.loc[:'20180101'] = historical_sp1.values
    sp1_series_cn = pd.Series(0, index=history_time)
    sp1_series_cn.loc[:'20180101'] = historical_sp1.values
    sp1_series_rw = pd.Series(0, index=history_time)
    sp1_series_rw.loc[:'20180101'] = historical_sp1.values
    spa_series = pd.DataFrame(0, index=history_time, columns=scraps)
    spa_series.loc[:'20180101']= historical_spa.values
    spa_series_cn = pd.DataFrame(0, index=history_time, columns=scraps)
    spa_series_cn.loc[:'20180101']= historical_spa.values
    spa_series_rw = pd.DataFrame(0, index=history_time, columns=scraps)
    spa_series_rw.loc[:'20180101']= historical_spa.values

    # Initialize mining stats
    mine_life_stats_panel_operating=mine_life_stats_panel_init(simulation_time, operating_mine_pool)
    mine_pool_new_last=pd.DataFrame()
    mine_life_stats_panel_new_last=pd.DataFrame()
    total_mining_prod=pd.Series(0, index=simulation_time)

    # Initilize sxew ids
    sxew_id_operating_bool=operating_mine_pool.loc[:, 'Payable percent (%)']==100
    sxew_id_operating=[i for i in sxew_id_operating_bool.index if sxew_id_operating_bool.loc[i]]
    conc_id_operating_bool=operating_mine_pool.loc[:, 'Payable percent (%)']!=100
    conc_id_operating=[i for i in conc_id_operating_bool.index if conc_id_operating_bool.loc[i]]
    sxew_id_new_bool=incentive_pool.loc[:, 'Payable percent (%)']==100
    sxew_id_new=[i for i in sxew_id_new_bool.index if sxew_id_new_bool.loc[i]]
    conc_id_new_bool=incentive_pool.loc[:, 'Payable percent (%)']!=100
    conc_id_new=[i for i in conc_id_new_bool.index if conc_id_new_bool.loc[i]]

    sxew_new=pd.Series(0, index=simulation_time)
    sxew_all=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    sxew_all_cn=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    sxew_all_rw=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    sxew_all.loc['20170101']=historical_prod.loc[2017, 'SX-EW production']
    sxew_all_cn.loc['20170101']=historical_prod_cn.loc[2017, 'SX-EW production']
    sxew_all_rw.loc['20170101']=historical_prod_rw.loc[2017, 'SX-EW production']


    # Initialize refinery stats
    ref_stats=ref_stats_init(simulation_time, ref_hyper_param)
    ref_stats_cn = ref_stats_init(simulation_time, ref_hyper_param_cn)
    ref_stats_rw = ref_stats_init(simulation_time, ref_hyper_param_rw)
    ref_bal_l1 = pd.Series(0, index = np.arange(2018,2041))
    ref_bal_l1_cn = pd.Series(0, index = np.arange(2018,2041))
    ref_bal_l1_rw = pd.Series(0, index = np.arange(2018,2041))

    # Initialize concentrate prod, add 2017
    conc_prod_series=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    conc_prod_series_cn=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    conc_prod_series_rw=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    # conc_prod_series.loc['20170101']=historical_prod.loc[2017, 'Concentrate production']
    conc_prod_series_cn.loc['20170101']=historical_prod_cn.loc[2017, 'Concentrate production']
    conc_prod_series_rw.loc['20170101']=historical_prod_rw.loc[2017, 'Concentrate production']
    conc_prod_series.loc['20170101'] = conc_prod_series_cn.loc['20170101'] + conc_prod_series_rw.loc['20170101']

    # Initialize refined supply and demand
    ref_prod_series=pd.Series(0, index=simulation_time)
    ref_prod_series_cn=pd.Series(0, index=simulation_time)
    ref_prod_series_rw=pd.Series(0, index=simulation_time)
    ref_demand_series=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    ref_demand_series_cn=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    ref_demand_series_rw=pd.Series(0, index=pd.date_range('20170101', simulation_end_time, freq='AS'))
    # ref_demand_series.loc['20170101']=historical_prod.loc[2017, 'Refined usage']
    ref_demand_series_cn.loc['20170101']=historical_prod_cn.loc[2017, 'Refined usage']# - historical_ref_imports_cn.loc[2017]
    ref_demand_series_rw.loc['20170101']=historical_prod_rw.loc[2017, 'Refined usage']# + historical_ref_imports_cn.loc[2017]
    ref_demand_series.loc['20170101'] = ref_demand_series_cn.loc['20170101'] + ref_demand_series_rw.loc['20170101']

    # Initialize end use by product stats
    use_product_future=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=use_product_history.columns)
    use_product_all_life=pd.concat([use_product_history, use_product_future])
    use_product_future_cn=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=use_product_history_cn.columns)
    use_product_all_life_cn=pd.concat([use_product_history_cn, use_product_future_cn])
    use_product_future_rw=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=use_product_history_rw.columns)
    use_product_all_life_rw=pd.concat([use_product_history_rw, use_product_future_rw])

    # Initialize old scrap history
    product_eol_history_cn=product_reach_eol(use_product_history_cn, product_lifetime_freq_df_cn)
    product_eol_history_rw=product_reach_eol(use_product_history_rw, product_lifetime_freq_df)
    product_eol_history=product_eol_history_cn + product_eol_history_rw
    # product_eol_history = product_reach_eol(use_product_history, product_lifetime_freq_df)
    waste_from_old_history_cn=product_eol_history_cn.apply(lambda x: (x*product_to_waste_collectable.T).sum(axis=1), axis=1).mul(sort_eff_cn).mul(collect_rate_cn)
    waste_from_old_history_rw=product_eol_history_rw.apply(lambda x: (x*product_to_waste_collectable.T).sum(axis=1), axis=1).mul(sort_eff).mul(collect_rate)
    # waste_from_old_history_cn=pd.DataFrame(np.matmul(product_eol_history_cn, product_to_waste_collectable), 
    #                                      index=product_eol_history_cn.index, 
    #                                      columns=product_to_waste_collectable.columns).mul(sort_eff_cn).mul(collect_rate_cn)
    # waste_from_old_history_rw=pd.DataFrame(np.matmul(product_eol_history_rw, product_to_waste_collectable), 
    #                                      index=product_eol_history_rw.index, 
    #                                      columns=product_to_waste_collectable.columns).mul(sort_eff).mul(collect_rate)
    # waste_from_old_history=pd.DataFrame(np.matmul(product_eol_history, product_to_waste_collectable), 
    #                                      index=product_eol_history.index, 
    #                                      columns=product_to_waste_collectable.columns).mul(sort_eff).mul(collect_rate)
    waste_from_old_history = waste_from_old_history_cn + waste_from_old_history_rw
    waste_from_old_future_cn=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_old_future_rw=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_old_future=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_old_all_life_cn=pd.concat([waste_from_old_history_cn, waste_from_old_future_cn])
    waste_from_old_all_life_rw=pd.concat([waste_from_old_history_rw, waste_from_old_future_rw])
    waste_from_old_all_life=pd.concat([waste_from_old_history, waste_from_old_future])

    # Old scrap available 
    old_scrap_available_history_cn = old_scrap_gen_init(product_eol_history_cn, product_to_waste_collectable, product_to_cathode_alloy,
                                        collect_rate_cn, sort_eff_cn, prod_spec.copy(), s2s, fraction_no1_old)
    old_scrap_available_history_rw = old_scrap_gen_init(product_eol_history_rw, product_to_waste_collectable, product_to_cathode_alloy,
                                        collect_rate, sort_eff, prod_spec.copy(), s2s, fraction_no1_old)
    # old_scrap_available_history = old_scrap_gen_init(product_eol_history, product_to_waste_collectable, product_to_cathode_alloy,
    #                                      collect_rate, sort_eff, prod_spec.copy(), s2s, fraction_no1 = 0.1)
    old_scrap_available_history = old_scrap_available_history_cn + old_scrap_available_history_rw

    old_scrap_available_future=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=old_scrap_available_history.columns)
    old_scrap_available_future_cn=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=old_scrap_available_history_cn.columns)
    old_scrap_available_future_rw=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=old_scrap_available_history_rw.columns)
    old_scrap_available_cn = pd.concat([old_scrap_available_history_cn, old_scrap_available_future_cn])
    old_scrap_available_rw = pd.concat([old_scrap_available_history_rw, old_scrap_available_future_rw])
    old_scrap_available = pd.concat([old_scrap_available_history, old_scrap_available_future])

    # Initialize direct melt demand, needed up here for scaling fruity alloyed
    direct_melt_sectorial_demand_cn=(use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Alloyed'])
    direct_melt_sectorial_demand_rw=(use_product_all_life_rw*product_to_cathode_alloy.loc[:, 'Alloyed'])
    direct_melt_sectorial_demand = direct_melt_sectorial_demand_cn + direct_melt_sectorial_demand_rw

    if include_unalloyed == 1:
    #     direct_melt_sectorial_demand.loc[:,'Unalloyed'] = (use_product_all_life*product_to_cathode_alloy.loc[:, 'Copper']).sum(axis = 1)
        direct_melt_sectorial_demand_cn.loc[:,'Unalloyed'] = (use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Copper']).sum(axis = 1)
        direct_melt_sectorial_demand_rw.loc[:,'Unalloyed'] = (use_product_all_life_rw*product_to_cathode_alloy.loc[:, 'Copper']).sum(axis = 1)
        direct_melt_sectorial_demand.loc[:, 'Unalloyed'] = direct_melt_sectorial_demand_cn.loc[:, 'Unalloyed'] + direct_melt_sectorial_demand_rw.loc[:, 'Unalloyed']
        # Including imports
        direct_melt_sectorial_demand_cn.loc[1960:2018, 'Unalloyed'] -= historical_ref_imports_cn.loc[1960:2018]
        direct_melt_sectorial_demand_rw.loc[1960:2018, 'Unalloyed'] += historical_ref_imports_cn.loc[1960:2018]
    #     direct_melt_sectorial_demand_cn.loc[:, 'Unalloyed'] = use_product_all_life_cn.sum(axis=1) - direct_melt_sectorial_demand_cn.sum(axis = 1)
    #     direct_melt_sectorial_demand_rw.loc[:, 'Unalloyed'] = direct_melt_sectorial_demand.loc[:,'Unalloyed'] - direct_melt_sectorial_demand_cn.loc[:,'Unalloyed']

    # Initialize new scrap history
    waste_from_new_history_cn=pd.DataFrame(0, index=waste_from_old_history_cn.index, columns=product_to_waste_collectable.columns)
    waste_from_new_history_rw=pd.DataFrame(0, index=waste_from_old_history_rw.index, columns=product_to_waste_collectable.columns)
    waste_from_new_history=pd.DataFrame(0, index=waste_from_old_history.index, columns=product_to_waste_collectable.columns)
    new_scrap_available_history_cn = pd.DataFrame(0, index = product_eol_history_cn.index, columns = old_scrap_available_history_cn.columns)
    new_scrap_available_history_rw = pd.DataFrame(0, index = product_eol_history_rw.index, columns = old_scrap_available_history_rw.columns)
    new_scrap_available_history = pd.DataFrame(0, index = product_eol_history.index, columns = old_scrap_available_history.columns)
    new_scrap_alloys_cn = pd.DataFrame(0, index = product_eol_history_cn.index, columns = list(prod_spec.loc[:,'Primary code'].unique())+list(['New No.1'])+list(fruity_alloyed.index))
    new_scrap_alloys_rw = pd.DataFrame(0, index = product_eol_history_rw.index, columns = list(prod_spec.loc[:,'Primary code'].unique())+list(['New No.1'])+list(fruity_alloyed.index))
    prod_spec_cop = prod_spec.copy()
    for i in prod_spec_cop.index:
        prod_spec_cop.loc[i+prod_spec.shape[0],:] = prod_spec_cop.loc[i,:]
        prod_spec_cop.loc[i+prod_spec.shape[0],'UNS'] = prod_spec_cop.loc[i,'UNS'] + '_rw'
    annual_recycled_content = pd.DataFrame(0, index = np.arange(1960,2041), columns = list(prod_spec_cop.loc[:,'UNS'])+list(fruity_alloys.index)+list(['Unalloyed CN', 'Unalloyed RoW', 'Secondary refined CN', 'Secondary refined RoW']))
    annual_recycled_volume = pd.DataFrame(0, index = np.arange(1960,2041), columns = list(prod_spec_cop.loc[:,'UNS'])+list(fruity_alloys.index)+list(['Unalloyed CN', 'Unalloyed RoW', 'Secondary refined CN', 'Secondary refined RoW']))

    for year_i in new_scrap_available_history.index:
        home_scrap_ratio=home_scrap_ratio_series.loc[year_i]
        exchange_scrap_ratio=exchange_scrap_ratio_series.loc[year_i]
        ######################################### unsure about this new_scrap_gen_cn and the fabrication efficiencies associated with it
        waste_from_new_year_i_cn=\
        simulate_new_scrap_one_year(year_i, use_product_history_cn, new_scrap_gen_cn, product_to_waste_no_loss, sort_eff_cn, 
                                    home_scrap_ratio, exchange_scrap_ratio)
        waste_from_new_history_cn.loc[year_i]=waste_from_new_year_i_cn.values
        waste_from_new_year_i_rw=\
        simulate_new_scrap_one_year(year_i, use_product_history_rw, new_scrap_gen, product_to_waste_no_loss, sort_eff, 
                                    home_scrap_ratio, exchange_scrap_ratio)
        waste_from_new_history_rw.loc[year_i]=waste_from_new_year_i_rw.values
    #     waste_from_new_year_i=\
    #     simulate_new_scrap_one_year(year_i, use_product_history, new_scrap_gen, product_to_waste_no_loss, sort_eff, 
    #                                 home_scrap_ratio, exchange_scrap_ratio)
        waste_from_new_history.loc[year_i]=waste_from_new_year_i_cn.values + waste_from_new_year_i_rw.values
        
        # Initialize new scrap availability history
        if pir_pcr == 0:
            new_scrap_available_year_i_cn = \
            new_scrap_gen_oneyear(use_product_history_cn.loc[year_i], product_to_waste_no_loss, product_to_cathode_alloy, 
                                collect_rate_cn, sort_eff_cn, prod_spec.copy(), s2s, new_scrap_gen_cn, exchange_scrap_ratio, 
                                home_scrap_ratio, fraction_no1_new)
            new_scrap_available_year_i_rw = \
            new_scrap_gen_oneyear(use_product_history_rw.loc[year_i], product_to_waste_no_loss, product_to_cathode_alloy, 
                                collect_rate, sort_eff, prod_spec.copy(), s2s, new_scrap_gen, exchange_scrap_ratio, 
                                home_scrap_ratio, fraction_no1_new)
        else:
            if fruity_multiplier != 0:
                fruity_alloyed.loc[:,'Quantity'] = fruity_multiplier*og_fruity_alloyed.loc[:,'Quantity']/direct_melt_sectorial_demand.loc[2018].sum()*direct_melt_sectorial_demand.loc[year_i].sum()
                while 0.3*direct_melt_sectorial_demand.loc[year_i,'Diverse'] < fruity_alloyed.loc[:,'Quantity'].sum():
                    fruity_alloyed.loc[:,'Quantity'] *= 0.5
                    print('New scrap alloyed reduced')
                
            new_scrap_available_year_i_cn, new_scrap_alloys_cn_year_i = \
            new_scrap_gen_oneyear(use_product_history_cn.loc[year_i], product_to_waste_no_loss, product_to_cathode_alloy, 
                                collect_rate_cn, sort_eff_cn, prod_spec.copy(), s2s, new_scrap_gen_cn, exchange_scrap_ratio, 
                                home_scrap_ratio, fraction_no1_new, pir_pcr, fruity_alloyed)
            new_scrap_available_year_i_rw, new_scrap_alloys_rw_year_i = \
            new_scrap_gen_oneyear(use_product_history_rw.loc[year_i], product_to_waste_no_loss, product_to_cathode_alloy, 
                                collect_rate, sort_eff, prod_spec.copy(), s2s, new_scrap_gen, exchange_scrap_ratio, 
                                home_scrap_ratio, fraction_no1_new, 2, fruity_alloyed)
            new_scrap_alloys_cn_year_i.loc[new_scrap_alloys_cn_year_i.loc[:,'Availability'] < 0,'Availability'] = 0
            new_scrap_alloys_rw_year_i.loc[new_scrap_alloys_rw_year_i.loc[:,'Availability'] < 0,'Availability'] = 0
            
        new_scrap_available_history_cn.loc[year_i] = new_scrap_available_year_i_cn
        new_scrap_available_history_rw.loc[year_i] = new_scrap_available_year_i_rw
        new_scrap_available_history.loc[year_i] = new_scrap_available_year_i_cn + new_scrap_available_year_i_rw
        if pir_pcr != 0:
            new_scrap_alloys_cn.loc[year_i] = new_scrap_alloys_cn_year_i.loc[:,'Availability']
            new_scrap_alloys_rw.loc[year_i] = new_scrap_alloys_rw_year_i.loc[:,'Availability']
        else:
            new_scrap_alloys_cn = 0
            new_scrap_alloys_rw = 0
    # waste_from_new_history = waste_from_new_history_cn + waste_from_new_history_rw
    # new_scrap_available_history = new_scrap_available_history_cn + new_scrap_available_history_rw
    if pir_pcr != 0:
        new_scrap_alloys_compositions = new_scrap_alloys_cn_year_i.loc[:,:'Low_Fe']
    else:
        new_scrap_alloys_compositions = 0

    waste_from_new_future_cn=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_new_future_rw=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_new_future=pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns=product_to_waste_collectable.columns)
    waste_from_new_all_life_cn=pd.concat([waste_from_new_history_cn, waste_from_new_future_cn])
    waste_from_new_all_life_rw=pd.concat([waste_from_new_history_rw, waste_from_new_future_rw])
    waste_from_new_all_life=pd.concat([waste_from_new_history, waste_from_new_future])
    new_scrap_available_future_cn = pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns = new_scrap_available_history_cn.columns)
    new_scrap_available_future_rw = pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns = new_scrap_available_history_rw.columns)
    new_scrap_available_future = pd.DataFrame(0, index=np.arange(2019, 2041, 1), columns = new_scrap_available_history.columns)
    new_scrap_available_cn = pd.concat([new_scrap_available_history_cn, new_scrap_available_future_cn])
    new_scrap_available_rw = pd.concat([new_scrap_available_history_rw, new_scrap_available_future_rw])
    new_scrap_available = pd.concat([new_scrap_available_history, new_scrap_available_future])

    # Assumes China scrap distribution is the same as the rest of the world
    waste_all_life_cn=waste_from_old_all_life_cn+waste_from_new_all_life_cn
    waste_all_life_rw=waste_from_old_all_life_rw+waste_from_new_all_life_rw
    waste_imports_cn_all = (waste_all_life_rw.apply(lambda x: x / waste_all_life_rw.sum(axis=1) * \
                            historical_prod_cn.loc[:,'Copper Scrap Imports, COMTRADE (kt)'])).loc[1960:2040]
    scrap_imports_cn_all = (old_scrap_available_rw.apply(lambda x: x / old_scrap_available_rw.sum(axis=1) * \
                            historical_prod_cn.loc[:,'Copper Scrap Imports, COMTRADE (kt)'])).loc[1960:2040]

    # China's future imports
    for i in waste_imports_cn_all.loc[2019:].index:
        waste_imports_cn_all.loc[i,:] = waste_imports_cn_all.loc[2018, :]
    for i in scrap_imports_cn_all.loc[2019:].index:
        scrap_imports_cn_all.loc[i, :] = scrap_imports_cn_all.loc[2018, :]
    og_scrap_imports_cn_all = scrap_imports_cn_all.copy()
        
    # Summing to produce all available scraps
    waste_all_life_cn += waste_imports_cn_all
    waste_all_life_rw -= waste_imports_cn_all
    waste_all_life = waste_all_life_cn + waste_all_life_rw
    all_scrap_available_cn = old_scrap_available_cn + new_scrap_available_cn + scrap_imports_cn_all
    all_scrap_available_rw = old_scrap_available_rw + new_scrap_available_rw - scrap_imports_cn_all
    all_scrap_available = all_scrap_available_cn + all_scrap_available_rw
    total_unalloyed_cn = all_scrap_available_cn.loc[:,'No.1':'No.2'].sum(axis = 1)
    total_unalloyed_rw = all_scrap_available_rw.loc[:,'No.1':'No.2'].sum(axis = 1)
    total_unalloyed = all_scrap_available.loc[:,'No.1':'No.2'].sum(axis = 1)

    # assuming historical No.1/No.2 ratio is at the 2018 value into the past
    if use_new_recovery == 0:
        all_scrap_available_cn.loc[:,'No.1'] = total_unalloyed_cn.loc[:] - historical_prod_cn.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff * 0.85
        all_scrap_available_rw.loc[:,'No.1'] = total_unalloyed_rw.loc[:] - historical_prod_rw.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff * 0.85
        # all_scrap_available.loc[:,'No.1'] = total_unalloyed.loc[:] - historical_prod.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff
        # all_scrap_available.loc[:,'No.2'] = historical_prod.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff
        all_scrap_available_cn.loc[:,'No.2'] = historical_prod_cn.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff * 0.85
        all_scrap_available_rw.loc[:,'No.2'] = historical_prod_rw.loc[:, 'Secondary refining production'] / scrap_to_cathode_eff * 0.85
    all_scrap_available.loc[:, 'No.1'] = all_scrap_available_cn.loc[:, 'No.1'] + all_scrap_available_rw.loc[:, 'No.1']
    all_scrap_available.loc[:, 'No.2'] = all_scrap_available_cn.loc[:, 'No.2'] + all_scrap_available_rw.loc[:, 'No.2']

    all_scrap_available_no_accumulation = all_scrap_available.copy() 
    all_scrap_available_no_accumulation_cn = all_scrap_available_cn.copy() 
    all_scrap_available_no_accumulation_rw = all_scrap_available_rw.copy() 
    # no_accumulation translates to the scrap entering the market that year - the increase in inventory that 
    # I'm thinking works as scrap supply for deterimining the supply-demand balance, while total_scrap_demand_all_life 
    # with scrappy=1 gives the scrap demand, or the scrap leaving inventory in that year

    # Removing new scrap that is now in alloy form
    if pir_pcr != 0:
    #     all_scrap_no_new_cn = all_scrap_available_cn.copy()
    #     all_scrap_no_new_rw = all_scrap_available_rw.copy()
    #     all_scrap_no_new_cn.loc[:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_cn, new_scrap_alloys_compositions)
    #     all_scrap_no_new_rw.loc[:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_rw, new_scrap_alloys_compositions)
    #     all_scrap_no_new = all_scrap_no_new_cn + all_scrap_no_new_rw
        all_scrap_available_cn.loc[1961:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_cn, new_scrap_alloys_compositions).loc[1961:]
        all_scrap_available_rw.loc[1961:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_rw, new_scrap_alloys_compositions).loc[1961:]
        all_scrap_available = all_scrap_available_cn + all_scrap_available_rw
        all_scrap_available_no_accumulation_cn.loc[1961:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_cn, new_scrap_alloys_compositions).loc[1961:]
        all_scrap_available_no_accumulation_rw.loc[1961:,'Al_Bronze':] -= alloy_to_scrap(new_scrap_alloys_rw, new_scrap_alloys_compositions).loc[1961:]
        all_scrap_available_no_accumulation = all_scrap_available_cn + all_scrap_available_rw


    # Initialize scrap demand
    # direct_melt_scrap_demand=(use_product_all_life*product_to_cathode_alloy.loc[:, 'Alloyed']).sum(axis=1)
    direct_melt_scrap_demand_cn=(use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Alloyed']).sum(axis=1)
    direct_melt_scrap_demand_rw=(use_product_all_life_rw*product_to_cathode_alloy.loc[:, 'Alloyed']).sum(axis=1)
    direct_melt_scrap_demand = direct_melt_scrap_demand_cn + direct_melt_scrap_demand_rw
    # direct_melt_scrap_demand_cn=(use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Alloyed']).sum(axis=1) * \
    #                             historical_prod_cn.loc[1960:, 'Direct melt'] / \
    #                             direct_melt_scrap_demand_cn.loc[1960:]
    # direct_melt_scrap_demand_rw = direct_melt_scrap_demand - direct_melt_scrap_demand_cn

    ## direct_melt_sectorial_demand=(use_product_all_life*product_to_cathode_alloy.loc[:, 'Alloyed'])
    # direct_melt_sectorial_demand_cn=(use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Alloyed'])
    # direct_melt_sectorial_demand_rw=(use_product_all_life_rw*product_to_cathode_alloy.loc[:, 'Alloyed'])
    # direct_melt_sectorial_demand = direct_melt_sectorial_demand_cn + direct_melt_sectorial_demand_rw
    ## direct_melt_sectorial_demand_cn=(use_product_all_life_cn*product_to_cathode_alloy.loc[:, 'Alloyed']).apply(lambda x: x* \
    ##                                 historical_prod_cn.loc[1960:, 'Direct melt'] / \
    ##                                direct_melt_scrap_demand_cn.loc[1960:])
    ## direct_melt_sectorial_demand_rw = direct_melt_sectorial_demand - direct_melt_sectorial_demand_cn

    if pir_pcr == 0:
        direct_melt_demand = pd.DataFrame(0, index = np.arange(1960,2041), columns = raw_price.columns)
        direct_melt_demand_cn = pd.DataFrame(0, index = np.arange(1960,2041), columns = raw_price.columns)
        direct_melt_demand_rw = pd.DataFrame(0, index = np.arange(1960,2041), columns = raw_price.columns)
    else:
        direct_melt_demand = pd.DataFrame(0, index = np.arange(1960,2041), columns = list(raw_price.columns) + list(new_scrap_alloys_compositions.index))
        direct_melt_demand_cn = pd.DataFrame(0, index = np.arange(1960,2041), columns = list(raw_price.columns) + list(new_scrap_alloys_compositions.index))
        direct_melt_demand_rw = pd.DataFrame(0, index = np.arange(1960,2041), columns = list(raw_price.columns) + list(new_scrap_alloys_compositions.index))
    fruity_detailed_demand = pd.DataFrame(0,index = pd.MultiIndex.from_product([direct_melt_demand_cn.index, \
                    fruity_alloys.index]), columns = direct_melt_demand.columns)

    # refined_scrap_demand=historical_prod.loc[:, 'Secondary refining production'].div(scrap_to_cathode_eff)
    refined_scrap_demand_cn=historical_prod_cn.loc[:, 'Secondary refining production'].div(scrap_to_cathode_eff)
    refined_scrap_demand_rw=historical_prod_rw.loc[:, 'Secondary refining production'].div(scrap_to_cathode_eff)
    refined_scrap_demand = refined_scrap_demand_cn + refined_scrap_demand_rw
    total_secondary_demand_all_life=pd.DataFrame({'Direct melt scrap': direct_melt_scrap_demand, 
                                            'Refined scrap': refined_scrap_demand})
    total_secondary_demand_all_life_cn=pd.DataFrame({'Direct melt scrap': direct_melt_scrap_demand_cn, 
                                            'Refined scrap': refined_scrap_demand_cn})
    total_secondary_demand_all_life_rw=pd.DataFrame({'Direct melt scrap': direct_melt_scrap_demand_rw, 
                                            'Refined scrap': refined_scrap_demand_rw})
    sec_ref_scrap_demand = direct_melt_demand.copy() # records scraps used in secondary refining
    sec_ref_scrap_demand_cn = direct_melt_demand.copy()
    sec_ref_scrap_demand_rw = direct_melt_demand.copy()
    fruity_demand = direct_melt_demand.copy()

    # Adding in historical blending
    # Generating Prices (assuming all metals' price ratios as compared with Ref_Cu remain constant at the earliest level we have)
    raw_price.loc[1960:1998, 'Ref_Cu'] = cathode_price_series.loc[datetime(1960, 1, 1):datetime(1998, 1, 1)].values
    raw_price.loc[1960,'Ref_Zn':'Cartridge'] = raw_price.loc[1999,'Ref_Zn':'Cartridge'] / raw_price.loc[1999,'Ref_Cu'] \
        * raw_price.loc[1960,'Ref_Cu']
    raw_price.loc[1960,'No.1':'No.2'] = raw_price.loc[1993,'No.1':'No.2'] / raw_price.loc[1993,'Ref_Cu'] \
        * raw_price.loc[1960,'Ref_Cu']
    raw_price.loc[1961:1998,'Ref_Zn':'Cartridge'] = raw_price.loc[1961:1998,'Ref_Cu'].apply(
        lambda x: x * raw_price.loc[1999,'Ref_Zn':'Cartridge'] / raw_price.loc[1999,'Ref_Cu'])
    raw_price.loc[1961:1992,'No.1':'No.2'] = raw_price.loc[1961:1992,'Ref_Cu'].apply(
        lambda x: x * raw_price.loc[1999,'No.1':'No.2'] / raw_price.loc[1999,'Ref_Cu'])
    raw_price_cn = raw_price.copy()
    raw_price_rw = raw_price.copy() # assuming that china's scrap prices may differ from those of the row due to the ban

    if 'Low_Brass' in all_scrap_available.columns:
        all_scrap_available.drop('Low_Brass', axis = 1, inplace = True)
        all_scrap_available_cn.drop('Low_Brass', axis = 1, inplace = True)
        all_scrap_available_rw.drop('Low_Brass', axis = 1, inplace = True)
    # Blending to determine demand in the direct melt sector
    for year_i in np.arange(1960,2019):
        if fruity_multiplier != 0:
            fruity_alloys.loc[:,'Quantity'] = fruity_multiplier*og_fruity_alloys.loc[:,'Quantity']/direct_melt_sectorial_demand.loc[2018].sum()*direct_melt_sectorial_demand.loc[year_i].sum()
        
        if pir_pcr != 0:
            if year_i > 1960:
                all_scrap_available_cn.loc[year_i,'Al_Bronze':] += alloy_to_scrap_1yr(new_scrap_alloys_cn.loc[year_i-1] - direct_melt_demand_cn.loc[year_i-1, new_scrap_alloys_cn.columns], new_scrap_alloys_compositions)
                all_scrap_available_rw.loc[year_i,'Al_Bronze':] += alloy_to_scrap_1yr(new_scrap_alloys_rw.loc[year_i-1] - direct_melt_demand_rw.loc[year_i-1, new_scrap_alloys_rw.columns], new_scrap_alloys_compositions)
    #             all_scrap_available_no_accumulation_cn.loc[year_i,'Al_Bronze':] += alloy_to_scrap_1yr(new_scrap_alloys_cn.loc[year_i-1] - direct_melt_demand_cn.loc[year_i-1, new_scrap_alloys_cn.columns], new_scrap_alloys_compositions)
    #             all_scrap_available_no_accumulation_rw.loc[year_i,'Al_Bronze':] += alloy_to_scrap_1yr(new_scrap_alloys_rw.loc[year_i-1] - direct_melt_demand_rw.loc[year_i-1, new_scrap_alloys_rw.columns], new_scrap_alloys_compositions)
    #             all_scrap_available_no_accumulation.loc[year_i] = all_scrap_available_no_accumulation_cn.loc[year_i] + all_scrap_available_no_accumulation_rw.loc[year_i]
            new_scrap_alloys_cn_year_i = new_scrap_alloys_compositions.copy().loc[:,'High_Cu':]
            new_scrap_alloys_cn_year_i.loc[:,'Availability'] = new_scrap_alloys_cn.loc[year_i]
            new_scrap_alloys_rw_year_i = new_scrap_alloys_compositions.copy().loc[:,'High_Cu':]
            new_scrap_alloys_rw_year_i.loc[:,'Availability'] = new_scrap_alloys_rw.loc[year_i]
            
        if year_i > 1960:   
    #         all_scrap_available.loc[year_i,:] += all_scrap_available.loc[year_i-1,:] - direct_melt_demand.loc[year_i-1,:]
            all_scrap_available_cn.loc[year_i,:] += all_scrap_available_cn.loc[year_i-1,:] - direct_melt_demand_cn.loc[year_i-1,:]
            all_scrap_available_rw.loc[year_i,:] += all_scrap_available_rw.loc[year_i-1,:] - direct_melt_demand_rw.loc[year_i-1,:]
            all_scrap_available.loc[year_i,:] = all_scrap_available_cn.loc[year_i,:] + all_scrap_available_rw.loc[year_i,:] 
        
        if rollover == 1:
            all_scrap_available_year_i = all_scrap_available.loc[year_i]
            all_scrap_available_year_i_cn = all_scrap_available_cn.loc[year_i]
            all_scrap_available_year_i_rw = all_scrap_available_rw.loc[year_i]
        else:
            all_scrap_available_year_i = all_scrap_available_no_accumulation.loc[year_i]
            all_scrap_available_year_i_cn = all_scrap_available_no_accumulation_cn.loc[year_i]
            all_scrap_available_year_i_rw = all_scrap_available_no_accumulation_rw.loc[year_i]
        if year_i < 1973:
            pir_price = 1.09
        else:
            pir_price = pir_price_set
        
        if (year_i == 1960 or slow_change == 0) and fraction_yellows != 0:
            direct_melt_demand_cn.loc[year_i,:], direct_melt_demand_rw.loc[year_i,:], sec_ref_scrap_demand_cn.loc[year_i,:], sec_ref_scrap_demand_rw.loc[year_i,:], fruity_detailed_demand.loc[idx[year_i,:],:], annual_recycled_content.loc[year_i], annual_recycled_volume.loc[year_i] = \
                blend_ff(all_scrap_available_year_i_cn, all_scrap_available_year_i_rw,
                                direct_melt_sectorial_demand_cn.loc[year_i], direct_melt_sectorial_demand_rw.loc[year_i],
                                raw_price_cn.loc[year_i], raw_price_rw.loc[year_i], s2s, prod_spec.copy(), raw_spec, 
                                refined_scrap_demand_cn.loc[year_i], refined_scrap_demand_rw.loc[year_i],
                                historical_prod_cn.loc[year_i,'Refined usage'], historical_prod_rw.loc[year_i,'Refined usage'],
                                fraction_yellows, unalloyed_tune, fruity_alloys, fruity_rr, pir_pcr, pir_fraction,
                                new_scrap_alloys_cn_year_i, new_scrap_alloys_rw_year_i, pir_price = pir_price)
            sec_ref_scrap_demand.loc[year_i,:] = sec_ref_scrap_demand_cn.loc[year_i,:] + sec_ref_scrap_demand_rw.loc[year_i,:]
            fruity_demand.loc[year_i,:] = fruity_detailed_demand.loc[idx[year_i,:],:].sum()

        elif year_i == 1960 or slow_change == 0:
    #         direct_melt_demand.loc[year_i,:] = pd.Series(blend(all_scrap_available_year_i, 
    #                                                        direct_melt_sectorial_demand.loc[year_i], 
    #                                                        raw_price.loc[year_i,:], s2s, prod_spec.copy(), raw_spec))
            direct_melt_demand_cn.loc[year_i,:], direct_melt_demand_rw.loc[year_i,:] = \
                                            pd.Series(blend_cn(all_scrap_available_year_i_cn, all_scrap_available_year_i_rw,
                                                            direct_melt_sectorial_demand_cn.loc[year_i], 
                                                            direct_melt_sectorial_demand_rw.loc[year_i],
                                                            raw_price_cn.loc[year_i,:], raw_price_rw.loc[year_i], s2s, prod_spec.copy(), raw_spec,
    #                                                        ref_demand = direct_melt_sectorial_demand_cn.loc[year_i,'Unalloyed']))
                                                            ref_demand_cn = ref_prod_history_cn.loc[year_i, 'Refined usage'],
                                                            ref_demand_rw = ref_prod_history_rw.loc[year_i, 'Refined usage']))
    #                                                        direct_melt_sectorial_demand_cn.loc[year_i,'Unalloyed'], fraction_yellows = fraction_yellows))
    #                                                            direct_melt_sectorial_demand_rw.loc[year_i,'Unalloyed'], fraction_yellows = fraction_yellows))
        else:
            direct_melt_demand.loc[year_i,:] = pd.Series(blend(all_scrap_available_year_i, 
                                                            direct_melt_sectorial_demand.loc[year_i], 
                                                            raw_price.loc[year_i,:], s2s, prod_spec.copy(), raw_spec, 
                                                            refined_scrap_demand.loc[year_i], ref_prod_history.loc[year_i,'Refined usage'],
                                                            fraction_yellows, unalloyed_tune, direct_melt_demand.loc[year_i-1]))
            direct_melt_demand_cn.loc[year_i,:] = pd.Series(blend(all_scrap_available_year_i_cn, 
                                                                direct_melt_sectorial_demand_cn.loc[year_i], 
                                                                raw_price_cn.loc[year_i,:], s2s, prod_spec.copy(), raw_spec,
                                                                refined_scrap_demand_cn.loc[year_i], ref_prod_history_cn.loc[year_i,'Refined usage'],
                                                                fraction_yellows, unalloyed_tune, direct_melt_demand_cn.loc[year_i-1]))
            direct_melt_demand_rw.loc[year_i,:] = pd.Series(blend(all_scrap_available_year_i_rw, 
                                                                direct_melt_sectorial_demand_rw.loc[year_i], 
                                                                raw_price_rw.loc[year_i,:], s2s, prod_spec.copy(), raw_spec,
                                                                refined_scrap_demand_rw.loc[year_i], ref_prod_history_rw.loc[year_i,'Refined usage'],
                                                                fraction_yellows, unalloyed_tune, direct_melt_demand_rw.loc[year_i-1]))
        direct_melt_demand.loc[year_i, :] = direct_melt_demand_cn.loc[year_i, :] + direct_melt_demand_rw.loc[year_i, :] 
        if fraction_yellows == 0:
            direct_melt_demand.loc[year_i,'No.2'] = refined_scrap_demand_cn[year_i] + refined_scrap_demand_rw[year_i]
            direct_melt_demand_cn.loc[year_i,'No.2'] = refined_scrap_demand_cn[year_i]
            direct_melt_demand_rw.loc[year_i,'No.2'] = refined_scrap_demand_rw[year_i]
        if year_i % 1 == 0:
            print(year_i)

    # if pir_pcr != 0:
    #     all_scrap_available_cn.loc[1961:,'Al_Bronze':] += alloy_to_scrap(new_scrap_alloys_cn, new_scrap_alloys_compositions).loc[1961:]
    #     all_scrap_available_rw.loc[1961:,'Al_Bronze':] += alloy_to_scrap(new_scrap_alloys_rw, new_scrap_alloys_compositions).loc[1961:]
    #     all_scrap_available = all_scrap_available_cn + all_scrap_available_rw
        
    if scrappy == 1:
        total_scrap_demand_all_life=pd.DataFrame({'Direct melt scrap': direct_melt_demand.loc[:,'Yellow_Brass':].sum(axis=1) - refined_scrap_demand, \
                                            'Refined scrap': refined_scrap_demand})
        total_scrap_demand_all_life_cn=pd.DataFrame({'Direct melt scrap': direct_melt_demand_cn.loc[:,'Yellow_Brass':].sum(axis=1) - refined_scrap_demand, \
                                            'Refined scrap': refined_scrap_demand_cn})
        total_scrap_demand_all_life_rw=pd.DataFrame({'Direct melt scrap': direct_melt_demand_rw.loc[:,'Yellow_Brass':].sum(axis=1) - refined_scrap_demand, \
                                            'Refined scrap': refined_scrap_demand_rw})
    elif scrappy == 2:
        total_scrap_demand_all_life=pd.DataFrame({'Direct melt scrap': direct_melt_demand.loc[:, scrap_subset].sum(axis=1) - refined_scrap_demand, \
                                                'Refined scrap': refined_scrap_demand})
        total_scrap_demand_all_life_cn=pd.DataFrame({'Direct melt scrap': direct_melt_demand_cn.loc[:, scrap_subset].sum(axis=1) - refined_scrap_demand_cn, \
                                                    'Refined scrap': refined_scrap_demand_cn})
        total_scrap_demand_all_life_rw=pd.DataFrame({'Direct melt scrap': direct_melt_demand_rw.loc[:, scrap_subset].sum(axis=1) - refined_scrap_demand_rw, \
                                                    'Refined scrap': refined_scrap_demand_rw})
    else:
        total_scrap_demand_all_life=pd.DataFrame({'Direct melt scrap': direct_melt_demand.sum(axis=1) - refined_scrap_demand, \
                                                'Refined scrap': refined_scrap_demand})
        total_scrap_demand_all_life_cn=pd.DataFrame({'Direct melt scrap': direct_melt_demand_cn.sum(axis=1) - refined_scrap_demand_cn, \
                                                    'Refined scrap': refined_scrap_demand_cn})
        total_scrap_demand_all_life_rw=pd.DataFrame({'Direct melt scrap': direct_melt_demand_rw.sum(axis=1) - refined_scrap_demand_rw, \
                                                    'Refined scrap': refined_scrap_demand_rw})

    scrap_use_avail_ratio = direct_melt_demand.loc[:,'Yellow_Brass':]/all_scrap_available.loc[1960:,:]
    scrap_use_avail_ratio_cn = direct_melt_demand_cn.loc[:,'Yellow_Brass':]/all_scrap_available_cn.loc[1960:,:]
    scrap_use_avail_ratio_rw = direct_melt_demand_rw.loc[:,'Yellow_Brass':]/all_scrap_available_rw.loc[1960:,:]
    if 'Low_Brass' in scrap_use_avail_ratio.columns:
        scrap_use_avail_ratio.drop('Low_Brass',axis=1,inplace=True)
        scrap_use_avail_ratio_cn.drop('Low_Brass',axis=1,inplace=True)
        scrap_use_avail_ratio_rw.drop('Low_Brass',axis=1,inplace=True)

    print('Historical initialization complete: '+str(datetime.now()))
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
idx=pd.IndexSlice


def integrand(x, mu, sigma, base):
    return norm.pdf(x, loc=mu, scale=sigma) * base**x


def integrand_exp(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma) * np.exp(x)


# ∆log(Intensity) = β0 × 1S + βS(∆log(Price)) × 1S + βGDP(∆log(GDPR)) × 1R 
def intensity_growth_sec_only(mu_int, sigma_int, 
                              mu_p, sigma_p, p_growth, 
                              mu_gdp, sigma_gdp, gdp_growth):
    # mu_int, sigma_int are the posteriors for intercept β0
    intercept_integral=integrate.quad(integrand_exp, -np.inf, np.inf, args=(mu_int, sigma_int))
    # p_growth=p_t/p_(t-1), where p_t is the trailing two year average first lag
    price_integral=integrate.quad(integrand, -np.inf, np.inf, args=(mu_p, sigma_p, p_growth))
    # gdp_growth similar to p_growth, not lagged
    gdp_integral=integrate.quad(integrand, -np.inf, np.inf, args=(mu_gdp, sigma_gdp, gdp_growth))
    # returns the expectation of intensity_growth=I_t/I(t-1)
    return intercept_integral[0] * price_integral[0] * gdp_integral[0] 


# ∆log(Intensity) = 0.5β0S × 1S + 0.5β0R × 1R + 0.5βS(∆log(Price)) × 1S + 0.5βR(∆log(Price)) × 1R + βGDP(∆log(GDPR)) × 1R
def intensity_growth_sec_reg(mu_int_sec, sigma_int_sec, mu_int_reg, sigma_int_reg, 
                             mu_p_sec, sigma_p_sec, mu_p_reg, sigma_p_reg, p_growth, 
                             mu_gdp, sigma_gdp, gdp_growth):
    # Intercept β0S and β0R
    intercept_integral_sec=integrate.quad(integrand_exp, -np.inf, np.inf, args=(0.5*mu_int_sec, 0.5*sigma_int_sec))
    intercept_integral_reg=integrate.quad(integrand_exp, -np.inf, np.inf, args=(0.5*mu_int_reg, 0.5*sigma_int_reg))
    # Price βS and βR
    price_integral_sec=integrate.quad(integrand, -np.inf, np.inf, args=(0.5*mu_p_sec, 0.5*sigma_p_sec, p_growth))
    price_integral_reg=integrate.quad(integrand, -np.inf, np.inf, args=(0.5*mu_p_reg, 0.5*sigma_p_reg, p_growth))
    # GDP βGDP
    gdp_integral=integrate.quad(integrand, -np.inf, np.inf, args=(mu_gdp, sigma_gdp, gdp_growth))
    return intercept_integral_sec[0] * intercept_integral_reg[0] * price_integral_sec[0] * price_integral_reg[0] * gdp_integral[0]


def intensity_growth_sec_only_specified(elas_sec_only, sec, reg, gdp_growth_reg, p_growth=1):
    mu_int=elas_sec_only.loc[sec, 'Intercept mean']
    sigma_int=elas_sec_only.loc[sec, 'Intercept SD']
    mu_p=elas_sec_only.loc[sec, 'Elasticity mean']
    sigma_p=elas_sec_only.loc[sec, 'Elasticity SD']
    
    mu_gdp=elas_sec_only.loc['GDP', 'Elasticity mean']
    sigma_gdp=elas_sec_only.loc['GDP', 'Elasticity SD']
    
    int_growth=intensity_growth_sec_only(mu_int, sigma_int, 
                                         mu_p, sigma_p, p_growth, 
                                         mu_gdp, sigma_gdp, gdp_growth_reg)
    return int_growth


def intensity_growth_sec_reg_specified(elas_sec_reg, sec, reg, gdp_growth_reg, p_growth=1):
    mu_int_sec=elas_sec_reg.loc[sec, 'Intercept mean']
    sigma_int_sec=elas_sec_reg.loc[sec, 'Intercept SD']
    mu_p_sec=elas_sec_reg.loc[sec, 'Elasticity mean']
    sigma_p_sec=elas_sec_reg.loc[sec, 'Elasticity SD']
    
    mu_int_reg=elas_sec_reg.loc[reg, 'Intercept mean']
    sigma_int_reg=elas_sec_reg.loc[reg, 'Intercept SD']
    mu_p_reg=elas_sec_reg.loc[reg, 'Elasticity mean']
    sigma_p_reg=elas_sec_reg.loc[reg, 'Elasticity SD']

    mu_gdp=elas_sec_reg.loc['GDP', 'Elasticity mean']
    sigma_gdp=elas_sec_reg.loc['GDP', 'Elasticity SD']
    
    int_growth=intensity_growth_sec_reg(mu_int_sec, sigma_int_sec, mu_int_reg, sigma_int_reg, 
                                        mu_p_sec, sigma_p_sec, mu_p_reg, sigma_p_reg, p_growth, 
                                        mu_gdp, sigma_gdp, gdp_growth_reg)
    return int_growth


def demand_prediction_mi(price_series, gdp_growth_prediction, intensity_init, volume_prediction, elas_mat, 
                         start_year=2015, end_year=2040, method='sec and reg',  verbose=0):
    price=price_series.copy()
    price.index=np.arange(1991,2041)
    
    regions=['China', 'EU', 'Japan', 'NAM', 'ROW']
    sectors=['Construction', 'Electrical', 'Industrial', 'Transport', 'Other']
    columns_mi=pd.MultiIndex.from_product([sectors, regions], names=['Sector', 'Region'])
    intensity_growth_prediction=pd.DataFrame(0,index=np.arange(start_year, end_year+1), columns=columns_mi)

    for t in intensity_growth_prediction.index:
        if verbose > 0:
            print('Calculating year: ', t)
        for sec in sectors:
            for reg in regions:
                gdp_growth=gdp_growth_prediction.loc[t, reg]+1
                p_growth=(price.loc[t-1]+price.loc[t-2])/(price.loc[t-2]+price.loc[t-3])
                if method == 'sec only':
                    int_growth=intensity_growth_sec_only_specified(elas_mat, sec, reg, gdp_growth, p_growth)
                elif method == 'sec and reg':
                    int_growth=intensity_growth_sec_reg_specified(elas_mat, sec, reg, gdp_growth, p_growth)
                intensity_growth_prediction.loc[t, idx[sec, reg]]=int_growth

    intensity_prediction=intensity_growth_prediction.cumprod(axis=0).mul(intensity_init)
    demand_prediction=intensity_prediction.mul(volume_prediction)
    
    return demand_prediction


def intensity_prediction_one_year(year_i, price_series, gdp_growth_prediction, intensity_last, volume_prediction, 
                               elas_mat, method='sec and reg'):

    t=pd.datetime(year_i, 1, 1)
    regions=['China', 'EU', 'Japan', 'NAM', 'ROW']
    sectors=['Construction', 'Electrical', 'Industrial', 'Transport', 'Other']
    columns_mi=pd.MultiIndex.from_product([sectors, regions], names=['Sector', 'Region'])
    intensity_growth_prediction=pd.DataFrame(0,index=[year_i], columns=columns_mi)
    method='sec and reg'

    for sec in sectors:
        for reg in regions:
            gdp_growth=gdp_growth_prediction.loc[year_i, reg]+1
            p_growth=(price_series.loc[t-relativedelta(years=1)]+price_series.loc[t-relativedelta(years=2)])\
                    /(price_series.loc[t-relativedelta(years=2)]+price_series.loc[t-relativedelta(years=3)])
            if method == 'sec only':
                int_growth=intensity_growth_sec_only_specified(elas_mat, sec, reg, gdp_growth, p_growth)
            elif method == 'sec and reg':
                int_growth=intensity_growth_sec_reg_specified(elas_mat, sec, reg, gdp_growth, p_growth)
            intensity_growth_prediction.loc[year_i, idx[sec, reg]]=int_growth
    
    intensity_prediction=intensity_growth_prediction.mul(intensity_last)

    return intensity_prediction



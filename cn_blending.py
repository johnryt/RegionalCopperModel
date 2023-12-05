# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:03:20 2017

@author: xinka
"""

import pandas as pd
import numpy as np
from gurobipy import *
from datetime import datetime

# old blender, new one is at the bottom
def blend_optimize(price, quantity, product_spec, CC, confidence, No2 = False):
    # Raw materials
    raw_spec = pd.read_excel("Data\\Raw_spec_201901.xlsx", index_col=0)
    raw_spec['cost'] = price
    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn,High_Ni, Low_Ni,\
    High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost= multidict(raw_spec.T.to_dict('list'))
    
    # Product specs
    element, max_spec, min_spec = multidict({
        "Cu":[product_spec.High_Cu,product_spec.Low_Cu],
        "Zn":[product_spec.High_Zn,product_spec.Low_Zn],
        "Pb":[product_spec.High_Pb,product_spec.Low_Pb],
        "Sn":[product_spec.High_Sn,product_spec.Low_Sn],
        "Ni":[product_spec.High_Ni,product_spec.Low_Ni],
        "Al":[product_spec.High_Al,product_spec.Low_Al],
        "Mn":[product_spec.High_Mn,product_spec.Low_Mn],
        "Fe":[product_spec.High_Fe,product_spec.Low_Fe]       
    })
    
    # Production quanity
    prod = quantity
    
    # Wrapup confidence
    s = confidence*2 - 1
    
    # Model
    blend_model = Model('Blending Starter')
    
    # Decision variables
    raw_demand = blend_model.addVars(raw, name="raw_demand")
    
    # Objective function
    blend_model.setObjective(raw_demand.prod(cost), GRB.MINIMIZE)
    
    # Specs constraints: if CC, use chance-constrained; else use deterministic
    if CC:
        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Cu[i] + Low_Cu[i])/2 + (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) <= max_spec['Cu'] * prod, "spec_Cu_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Cu[i] + Low_Cu[i])/2 - (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) >= min_spec['Cu'] * prod, "spec_Cu_lo")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Zn[i] + Low_Zn[i])/2 + (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) <= max_spec['Zn'] * prod, "spec_Zn_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Zn[i] + Low_Zn[i])/2 - (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) >= min_spec['Zn'] * prod, "spec_Zn_lo")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Pb[i] + Low_Pb[i])/2 + (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) <= max_spec['Pb'] * prod, "spec_Pb_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Pb[i] + Low_Pb[i])/2 - (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) >= min_spec['Pb'] * prod, "spec_Pb_lo")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Sn[i] + Low_Sn[i])/2 + (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) <= max_spec['Sn'] * prod, "spec_Sn_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Sn[i] + Low_Sn[i])/2 - (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) >= min_spec['Sn'] * prod, "spec_Sn_lo")        

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Ni[i] + Low_Ni[i])/2 + (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) <= max_spec['Ni'] * prod, "spec_Ni_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Ni[i] + Low_Ni[i])/2 - (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) >= min_spec['Ni'] * prod, "spec_Ni_lo")        

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Al[i] + Low_Al[i])/2 + (High_Al[i] - Low_Al[i])/2*s) for i in raw) <= max_spec['Al'] * prod, "spec_Al_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Al[i] + Low_Al[i])/2 - (High_Al[i] - Low_Al[i])/2*s) for i in raw) >= min_spec['Al'] * prod, "spec_Al_lo")        

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Mn[i] + Low_Mn[i])/2 + (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) <= max_spec['Mn'] * prod, "spec_Mn_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Mn[i] + Low_Mn[i])/2 - (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) >= min_spec['Mn'] * prod, "spec_Mn_lo")        

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Fe[i] + Low_Fe[i])/2 + (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) <= max_spec['Fe'] * prod, "spec_Fe_up")

        blend_model.addConstr(
                quicksum(raw_demand[i] * ((High_Fe[i] + Low_Fe[i])/2 - (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) >= min_spec['Fe'] * prod, "spec_Fe_lo")        
        
        
    else:
        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Cu[i] + Low_Cu[i])/2 for i in raw), min_spec['Cu'] * prod, max_spec['Cu'] * prod, "spec_Cu")
    
        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Zn[i] + Low_Zn[i])/2 for i in raw), min_spec['Zn'] * prod, max_spec['Zn'] * prod, "spec_Zn")
    
        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Pb[i] + Low_Pb[i])/2 for i in raw), min_spec['Pb'] * prod, max_spec['Pb'] * prod, "spec_Pb")
    
        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Sn[i] + Low_Sn[i])/2 for i in raw), min_spec['Sn'] * prod, max_spec['Sn'] * prod, "spec_Sn")

        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Ni[i] + Low_Ni[i])/2 for i in raw), min_spec['Ni'] * prod, max_spec['Ni'] * prod, "spec_Ni")

        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Al[i] + Low_Al[i])/2 for i in raw), min_spec['Al'] * prod, max_spec['Al'] * prod, "spec_Al")

        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Mn[i] + Low_Mn[i])/2 for i in raw), min_spec['Mn'] * prod, max_spec['Mn'] * prod, "spec_Mn")

        blend_model.addRange(
                quicksum(raw_demand[i] * (High_Fe[i] + Low_Fe[i])/2 for i in raw), min_spec['Fe'] * prod, max_spec['Fe'] * prod, "spec_Fe")
    
    
    # Mass constraint
    blend_model.addConstr(
        quicksum(raw_demand[i] for i in raw) >= prod, "mass")
    
    
    if No2 == False:
        blend_model.addConstr(
                raw_demand['No.2'] == 0, "no #2")
    
    
    # Optimize!
    blend_model.update()
    blend_model.setParam( 'OutputFlag', False )
    blend_model.optimize()
    
    # Return results
    demand = blend_model.getAttr('x', raw_demand)
    return(demand)

# Defines the average cost curves for a given annual price and availability, generated originally from China import data to give the curve shape but otherwise based on assumptions that the curve average will be near the annual price, which is relatively true for the small amount of data we have. Original function in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\00 Simulation\06 Module integration\Other scenarios\China import ban\New Blending\New Blending.ipynb
def avg_cost_curve(x, price, avail, flag=-1): # b for breaks, s for slopes
    '''takes the quantity desired as x, average price for scrap type, and availability of that scrap type'''
    # b for breaks, s for slopes
    old_b = np.array([0, 0.97692502, 16.53142747, 71.65446698, 88.26814068])
    old_s = np.array([5244.55260855, 57.77448861, 20.35459672, 38.55811168])
    avail_scale = 88.2681406817016
    price_scale = 5868.547548553173
    # The above values all come from the No 2 scrap average cost curve developed in Order book initialization.ipynb
    b = old_b*avail/avail_scale
    s = old_s*price/price_scale/avail*avail_scale*0.9004710323359906 # extra ~0.9 scaling factor comes from intercept correction so that we don't have negative values
    if flag == 0:
        condlist = [x < b[1], (x >= b[1]) & (x < b[2]), (x >= b[2]) & (x < b[3]), (x >= b[3])]#, x >= b[4]]
        funclist = [lambda x: s[0]*x, lambda x: s[0]*b[1] + s[1]*(x-b[1]), lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(x - b[2]), 
                   lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(b[3] - b[2]) + s[3]*(x-b[3])] 
                   #lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(b[3] - b[2]) + s[3]*(b[4]-b[3]) + 100*s[0]*(x-b[4])]
    elif flag > 0:
        condlist = [x >= b[0]]
#         funclist = [lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*x]
        funclist = [lambda x: price + flag*x]  
    elif flag == -1:
        condlist = [x < b[2], (x >= b[2]) & (x < b[3]), (x >= b[3]) & (x < b[4]), x >= b[4]]
        funclist = [lambda x: s[0]*b[1] + s[1]*(x-b[1]), lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(x - b[2]), 
                   lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(b[3] - b[2]) + s[3]*(x-b[3]),
                   lambda x: s[0]*b[1] + s[1]*(b[2]-b[1]) + s[2]*(b[3] - b[2]) + s[3]*(b[4]-b[3]) + 100*s[0]*(x-b[4])]
    return np.piecewise(x, condlist, funclist)

# Converts from sectorial quantities to alloy quantities. Original in: C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\00 Simulation\06 Module integration\Other scenarios\China import ban\Order book initialization\Order book initialization.ipynb
def to_alloy(sectorial, s2s, prod_spec):
    # Convert to shapes (only for alloyed scrap):
    shapes = pd.Series(0,index = s2s.index)
    shapes = sectorial.dot(s2s.div(s2s.sum(axis=0)).fillna(0).transpose())
    if 'Unalloyed' in sectorial.index or 'Unalloyed' in prod_spec.index:
        prod_spec.drop('Unalloyed',inplace=True)
    # Go from shape quantity to scrap & alloy quantity:
    prod_spec.loc[:,'Quantity'] = 0
    for i in shapes.index:
        prod_spec.loc[prod_spec.loc[:,'Category']==i,'Quantity'] = prod_spec.loc[prod_spec.loc[:,'Category']==i,\
                                            'Fraction']*shapes.loc[i]

    return prod_spec.loc[:,'Quantity'] # for individual alloys




# Original function in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\00 Simulation\06 Module integration\Other scenarios\China import ban\New Blending\New Blending.ipynb
def blend(availability, alloy_demand, raw_price, s2s, prod_spec, raw_spec, sec_ref_prod = 0, ref_demand = 0, fraction_yellows = 0, unalloyed_tune = 1):
    '''Outputs demand for each scrap grade given demand for unalloyed and alloyed shapes for semis. With sec_ref_prod 
    specified, includes secondary refined production in the blending optimization, where secondary refineries can consume all
    scrap types except No.1, although yellow brasses are limited in refined use (based on USGS [values] and SMM [checkmarks])
    using the fraction_yellows variable. Giving ref_demand causes weighting within the optimization to put Ref_Cu consumption 
    closer to that value, where both No.1 and No.2 availabilities were set to the total unalloyed value and unalloyed_tune
    multiplies by that value to change unalloyed scrap use. '''

    prod_spec = prod_spec.copy()
    alloy_demand = alloy_demand.copy()
    raw_spec['Price'] = raw_price
    raw_spec['Availability'] = availability
    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn,High_Ni, Low_Ni,\
        High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost, avail = multidict(raw_spec.T.to_dict('list'))

    unalloyed_quant = 0
    if 'Unalloyed' in alloy_demand.index:
        unalloyed_quant = alloy_demand.copy().loc['Unalloyed']
        alloy_demand.drop('Unalloyed',inplace = True)

    production = list(to_alloy(alloy_demand.copy(),s2s,prod_spec.copy()))

    if unalloyed_quant != 0: 
        prod_spec.loc['Unalloyed','High_Cu':'Low_Fe'] = pd.Series([100, 99.8, 0.01, 0, 0.01, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0], index = prod_spec.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec.loc['Unalloyed','UNS'] = 'Unalloyed'
        prod_spec.loc['Unalloyed','Alloy Type':'Category'] = prod_spec.iloc[0,:].loc['Alloy Type':'Category']
        production.append(unalloyed_quant)

    if fraction_yellows != 0:
        production.append(sec_ref_prod)
        prod_spec.loc['Secondary refined', 'High_Cu':'Low_Fe'] = pd.Series([100, 50, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0], index = prod_spec.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec.loc['Secondary refined', 'UNS'] = 'Secondary refined'
        prod_spec.loc['Secondary refined', 'Alloy Type':'Category'] = prod_spec.iloc[0,:].loc['Alloy Type':'Category']
        avail['No.1'] = unalloyed_tune*(avail['No.1'] + avail['No.2'])
        avail['No.2'] = avail['No.1']

    # Product specs
    element, max_spec, min_spec = multidict({
        "Cu":[list(prod_spec.High_Cu.values),list(prod_spec.Low_Cu.values)],
        "Zn":[list(prod_spec.High_Zn.values),list(prod_spec.Low_Zn.values)],
        "Pb":[list(prod_spec.High_Pb.values),list(prod_spec.Low_Pb.values)],
        "Sn":[list(prod_spec.High_Sn.values),list(prod_spec.Low_Sn.values)],
        "Ni":[list(prod_spec.High_Ni.values),list(prod_spec.Low_Ni.values)],
        "Al":[list(prod_spec.High_Al.values),list(prod_spec.Low_Al.values)],
        "Mn":[list(prod_spec.High_Mn.values),list(prod_spec.Low_Mn.values)],
        "Fe":[list(prod_spec.High_Fe.values),list(prod_spec.Low_Fe.values)]       
    })

    alloys = list(prod_spec.UNS.values)
    scraps = sorted(prod_spec.loc[:,'Alloy Type'].unique())

    num_alloys = len(alloys)
    alloy_blend = alloys[0:num_alloys]
    scraps = list(['Al_Bronze', 'Cartridge', 'Mn_Bronze', 'Ni_Ag', 'No.1', 'No.2', 'Ocean', 'Pb_Sn_Bronze', 'Pb_Yellow_Brass', 'Red_Brass', 'Sn_Bronze', 'Yellow_Brass'])
    refs = list(['Ref_Cu', 'Ref_Al', 'Ref_Fe', 'Ref_Mn', 'Ref_Ni', 'Ref_Pb', 'Ref_Sn', 'Ref_Zn'])

    confidence = 0.95
    s = confidence*2 - 1
    CC = True

    m = Model('Blending')

    raw_demand = m.addVars(alloy_blend, raw, name='raw_demand')

    for a_i in range(0,num_alloys):
        a = alloy_blend[a_i]
        # Specs constraints: if CC, use chance-constrained; else use deterministic
        if CC:
            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 + (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) <= max_spec['Cu'][a_i] * production[a_i], "spec_Cu_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 - (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) >= min_spec['Cu'][a_i] * production[a_i], "spec_Cu_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 + (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) <= max_spec['Zn'][a_i] * production[a_i], "spec_Zn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 - (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) >= min_spec['Zn'][a_i] * production[a_i], "spec_Zn_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 + (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) <= max_spec['Pb'][a_i] * production[a_i], "spec_Pb_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 - (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) >= min_spec['Pb'][a_i] * production[a_i], "spec_Pb_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 + (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) <= max_spec['Sn'][a_i] * production[a_i], "spec_Sn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 - (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) >= min_spec['Sn'][a_i] * production[a_i], "spec_Sn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 + (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) <= max_spec['Ni'][a_i] * production[a_i], "spec_Ni_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 - (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) >= min_spec['Ni'][a_i] * production[a_i], "spec_Ni_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 + (High_Al[i] - Low_Al[i])/2*s) for i in raw) <= max_spec['Al'][a_i] * production[a_i], "spec_Al_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 - (High_Al[i] - Low_Al[i])/2*s) for i in raw) >= min_spec['Al'][a_i] * production[a_i], "spec_Al_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 + (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) <= max_spec['Mn'][a_i] * production[a_i], "spec_Mn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 - (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) >= min_spec['Mn'][a_i] * production[a_i], "spec_Mn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 + (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) <= max_spec['Fe'][a_i] * production[a_i], "spec_Fe_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 - (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) >= min_spec['Fe'][a_i] * production[a_i], "spec_Fe_lo")        
        else: 
            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Cu[i] + Low_Cu[i])/2 for i in raw), min_spec['Cu'][a_i] * production[a_i], max_spec['Cu'][a_i] * production[a_i], "spec_Cu")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Zn[i] + Low_Zn[i])/2 for i in raw), min_spec['Zn'][a_i] * production[a_i], max_spec['Zn'][a_i] * production[a_i], "spec_Zn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Pb[i] + Low_Pb[i])/2 for i in raw), min_spec['Pb'][a_i] * production[a_i], max_spec['Pb'][a_i] * production[a_i], "spec_Pb")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Sn[i] + Low_Sn[i])/2 for i in raw), min_spec['Sn'][a_i] * production[a_i], max_spec['Sn'][a_i] * production[a_i], "spec_Sn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Ni[i] + Low_Ni[i])/2 for i in raw), min_spec['Ni'][a_i] * production[a_i], max_spec['Ni'][a_i] * production[a_i], "spec_Ni")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Al[i] + Low_Al[i])/2 for i in raw), min_spec['Al'][a_i] * production[a_i], max_spec['Al'][a_i] * production[a_i], "spec_Al")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Mn[i] + Low_Mn[i])/2 for i in raw), min_spec['Mn'][a_i] * production[a_i], max_spec['Mn'][a_i] * production[a_i], "spec_Mn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Fe[i] + Low_Fe[i])/2 for i in raw), min_spec['Fe'][a_i] * production[a_i], max_spec['Fe'][a_i] * production[a_i], "spec_Fe")

        # Mass constraint
        m.addConstr(quicksum(raw_demand[(a,i)] for i in raw) >= production[a_i], "mass")

        # No No.2
        if a == 'Secondary refined':
            for rs in refs + list(['No.1']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no #1 or refined")
            for rs in ['Yellow_Brass', 'Pb_Yellow_Brass', 'Cartridge']:
                m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail[rs], "Low yellows")
        elif a == 'Unalloyed':
            for rs in refs[1:] + list(['No.2']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no refined aside from Cu")
        else:
            m.addConstr(raw_demand[(a,'No.2')] == 0, "no #2")

    new_quant = {}
    new_cost = {}
    n = 1000
    slope = -1
    width = sum(production[0:num_alloys])/n
    raw_demand_tot = m.addVars(raw,name='raw_demand_tot')
    if fraction_yellows != 0:
        m.addConstr(raw_demand_tot['No.1'] + raw_demand_tot['No.2'] <= avail['No.1'], 'unalloyed constraint') # since avail['No.1'] == avail['No.2'] == total unalloyed avail
#     m.addConstr(raw_demand_tot['Ref_Cu'] > 0.0, 'Ref_Cu > 0')
    for scrap in raw:
        m.addConstr(quicksum(raw_demand[(a,scrap)] for a in alloy_blend) == raw_demand_tot[scrap])

        new_quant.update({scrap: np.linspace(0,sum(production),n)})
        if scrap in scraps:
            new_cost.update({scrap: np.cumsum(width*avg_cost_curve(new_quant[scrap],cost[scrap],avail[scrap],slope))}) # effectively integrating under the curve
            m.addConstr(raw_demand_tot[scrap] <= avail[scrap])
        elif scrap == 'Ref_Cu' and fraction_yellows != 0:
            new_cost.update({scrap: cost['Ref_Cu']*abs(ref_demand - new_quant['Ref_Cu'])**2/(ref_demand)**2*
                             new_quant['Ref_Cu'] + cost['Ref_Cu']*new_quant['Ref_Cu']})
        else:
            new_cost.update({scrap: cost[scrap]*new_quant[scrap]})
        m.setPWLObj(raw_demand_tot[scrap],new_quant[scrap],new_cost[scrap])


    # Optimize!
    m.update()
    m.setParam( 'OutputFlag', False )
    m.optimize()

    # Return results
    demand = pd.Series(m.getAttr('x', raw_demand_tot))
    if fraction_yellows != 0:
        idx = pd.IndexSlice
        refined_secondary_demand = pd.Series(m.getAttr('x', raw_demand)).loc[idx['Secondary refined',:]]
        return demand, refined_secondary_demand
    else:
        return demand
    # print(n)
    
def system_initialization(og_fruity_alloys):
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
    fruity_multiplier = 1 #= 0.01*30997.694215369822/og_fruity_alloys.loc[:,'Quantity'].sum() # 30997 is 2018 direct_melt_sectorial_demand value, the leading value (eg 0.01) would be the fraction of market occupied by these alloys
    pir_pcr = 1
    pir_fraction = -1
    fruity_rr = [0.01,0.01,0.01,0.01]
    scrap_bal_correction = 0.928

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
    waste_from_old_history_cn=pd.DataFrame(np.matmul(product_eol_history_cn, product_to_waste_collectable), 
                                         index=product_eol_history_cn.index, 
                                         columns=product_to_waste_collectable.columns).mul(sort_eff_cn).mul(collect_rate_cn)
    waste_from_old_history_rw=pd.DataFrame(np.matmul(product_eol_history_rw, product_to_waste_collectable), 
                                         index=product_eol_history_rw.index, 
                                         columns=product_to_waste_collectable.columns).mul(sort_eff).mul(collect_rate)
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

    # Initialize new scrap history
    waste_from_new_history_cn=pd.DataFrame(0, index=waste_from_old_history_cn.index, columns=product_to_waste_collectable.columns)
    waste_from_new_history_rw=pd.DataFrame(0, index=waste_from_old_history_rw.index, columns=product_to_waste_collectable.columns)
    waste_from_new_history=pd.DataFrame(0, index=waste_from_old_history.index, columns=product_to_waste_collectable.columns)
    new_scrap_available_history_cn = pd.DataFrame(0, index = product_eol_history_cn.index, columns = old_scrap_available_history_cn.columns)
    new_scrap_available_history_rw = pd.DataFrame(0, index = product_eol_history_rw.index, columns = old_scrap_available_history_rw.columns)
    new_scrap_available_history = pd.DataFrame(0, index = product_eol_history.index, columns = old_scrap_available_history.columns)
    new_scrap_alloys_cn = pd.DataFrame(0, index = product_eol_history_cn.index, columns = list(prod_spec.loc[:,'Primary code'].unique())+list(fruity_alloyed.index))
    new_scrap_alloys_rw = pd.DataFrame(0, index = product_eol_history_rw.index, columns = list(prod_spec.loc[:,'Primary code'].unique())+list(fruity_alloyed.index))

    globals().update(locals())
    print(str(datetime.now()))

def blend_ff(availability_cn, availability_rw, alloy_demand_cn, alloy_demand_rw, raw_price_cn, raw_price_rw, s2s, 
             prod_spec, raw_spec, sec_ref_prod_cn = 0, sec_ref_prod_rw = 0, ref_demand_cn = 0, ref_demand_rw = 0, 
             fraction_yellows = 0, unalloyed_tune = 1, fruity_alloys=0, fruity_rr = [0,0,0,0], pir_pcr = 0, 
             pir_fraction = -1, new_scrap_alloys_cn = 0, new_scrap_alloys_rw = 0, pir_price = 0.9, cn_scrap_imports = 0,
             rc_variance = 0, og_rc = 0):
    '''Outputs demand for each scrap grade given demand for unalloyed and alloyed shapes for semis. With sec_ref_prod 
    specified, includes secondary refined production in the blending optimization, where secondary refineries can consume all
    scrap types except No.1, although yellow brasses are limited in refined use (based on USGS [values] and SMM [checkmarks])
    using the fraction_yellows variable. Giving ref_demand causes weighting within the optimization to put Ref_Cu consumption 
    closer to that value, where both No.1 and No.2 availabilities were set to the total unalloyed value and unalloyed_tune
    multiplies by that value to change unalloyed scrap use. '''
    
#     pir_pcr = 0 # zero excludes pir consideration, 1 allows new scrap from that year to be used
#     pir_fraction = -1 # set to zero to keep fruity from using PIR, -1 for no constraints on fruity PIR content
    weight_cu = 0
    multiplier = 1.5
    wiggle = 0.01
#     pir_price = #1.09 previously 1.09, but resulted in very little PIR consumption - going with 0.65 since it shows ~95% of PIR consumed, which feels closer to right and allows for increases and decreases as price changes
#     rc_variance = 0.95
    
    scraps = list(['Al_Bronze', 'Cartridge', 'Mn_Bronze', 'Ni_Ag', 'No.1', 'No.2', 'Ocean', 'Pb_Sn_Bronze', 'Pb_Yellow_Brass', 'Red_Brass', 'Sn_Bronze', 'Yellow_Brass'])
    refs = list(['Ref_Cu', 'Ref_Al', 'Ref_Fe', 'Ref_Mn', 'Ref_Ni', 'Ref_Pb', 'Ref_Sn', 'Ref_Zn'])

    prod_spec1 = prod_spec.copy()
    num_prod = prod_spec1.shape[0]
    alloy_demand_cn1 = alloy_demand_cn.copy()
    alloy_demand_rw1 = alloy_demand_rw.copy()
    new_scrap_alloys_cn1 = new_scrap_alloys_cn.copy()
    new_scrap_alloys_rw1 = new_scrap_alloys_rw.copy()
    if type(fruity_alloys) != type(0):
        fruity_alloys1 = fruity_alloys.copy()
    raw_spec_cn1 = raw_spec.copy()
    raw_spec_rw1 = raw_spec.copy()
    raw_spec_cn1['Price'] = raw_price_cn
    raw_spec_rw1['Price'] = raw_price_rw
    if type(cn_scrap_imports) != type(0):
        availabliity_cn -= cn_scrap_imports.loc[:,'Availability']
    raw_spec_cn1['Availability'] = availability_cn
    raw_spec_rw1['Availability'] = availability_rw
    if pir_pcr == 1:
#         for sc in scraps: # this is from a previous attempt where each new scrap's price was determined relative to its corresponding scrap type - it appears that historical consistency aligns with it corresponding with No.1 instead
#             new_scrap_alloys_cn1.loc[new_scrap_alloys_cn1.loc[:,'Alloy Type']==sc,'Price'] = raw_price_cn.loc[sc] * pir_price
#             new_scrap_alloys_rw1.loc[new_scrap_alloys_rw1.loc[:,'Alloy Type']==sc,'Price'] = raw_price_rw.loc[sc] * pir_price
#         new_scrap_alloys_cn1.drop(inplace=True, columns=['Alloy Type'])
#         new_scrap_alloys_rw1.drop(inplace=True, columns=['Alloy Type']) # had also required changing the new_scrap_alloys_rw_year_i = new_scrap_alloys_compositions.copy().loc[:,'High_Cu':] from 'Ref_Cu': to 'Alloy Type':
        new_scrap_alloys_cn1.loc[:, 'Price'] = raw_price_cn.loc['No.1'] * pir_price
        new_scrap_alloys_rw1.loc[:, 'Price'] = raw_price_rw.loc['No.1'] * pir_price
        raw_spec_cn1 = pd.concat([raw_spec_cn1,new_scrap_alloys_cn1], sort=False)
        raw_spec_rw1 = pd.concat([raw_spec_rw1,new_scrap_alloys_rw1], sort=False)
    if type(cn_scrap_imports) != type(0):
        cn_scrap_imports['Price'] = raw_price_cn
        rw_scrap_imports = cn_scrap_imports.copy()
        rw_scrap_imports.loc[:,'Availability'] = 0
        raw_spec_cn1 = pd.concat([raw_spec_cn1, cn_scrap_imports], sort=False)
        raw_spec_rw1 = pd.concat([raw_spec_cn1, rw_scrap_imports], sort=False)

    if 'Alloy Type' in raw_spec_cn1.columns:
        raw_spec_cn1.drop(columns='Alloy Type', inplace=True)
    if 'Alloy Type' in raw_spec_rw1.columns:
        raw_spec_rw1.drop(columns='Alloy Type', inplace=True)
    
    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn, High_Ni, Low_Ni,\
        High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost_cn, avail_cn = multidict(raw_spec_cn1.T.to_dict('list'))
    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn, High_Ni, Low_Ni,\
        High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost_rw, avail_rw = multidict(raw_spec_rw1.T.to_dict('list'))
    
    if type(fruity_alloys) != type(0):
        if 'Unalloyed' in alloy_demand_cn1.index:
            while 0.8*alloy_demand_cn1.loc['Unalloyed'] < fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1','Quantity'].sum():
                fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1','Quantity'] *= 0.5
                print('Unalloyed reduced')
            alloy_demand_cn1.loc['Unalloyed'] -= fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1','Quantity'].sum()
        if fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum() < alloy_demand_cn1.loc['Electronic']:
            alloy_demand_cn1.loc['Electronic'] -= fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
#         elif fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum() < alloy_demand_cn1.loc['Diverse']:
#             alloy_demand_cn1.loc['Electronic'] -= 0.1*fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
#             alloy_demand_cn1.loc['Diverse'] -= 0.9*fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
        else:
            bothsies = alloy_demand_cn1.loc['Diverse']+alloy_demand_cn1.loc['Electronic']+alloy_demand_cn1.loc['Consumer']
            balancey = bothsies - fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
            while balancey + 0.7*alloy_demand_cn1.loc['Electrical Automotive'] < 0:
                balancey = bothsies - fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
                fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'] *= 0.5
                print('Alloyed reduced')
            alloy_demand_cn1.loc['Electronic'] -= alloy_demand_cn1.loc['Electronic']/bothsies*fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
            alloy_demand_cn1.loc['Diverse'] -= alloy_demand_cn1.loc['Diverse']/bothsies*fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
            alloy_demand_cn1.loc['Consumer'] -= alloy_demand_cn1.loc['Consumer']/bothsies*fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()
            if balancey < 0:
                alloy_demand_cn1.loc['Electrical Automotive'] += balancey
    unalloyed_quant_cn = 0
    unalloyed_quant_rw = 0
    if 'Unalloyed' in alloy_demand_cn1.index:
        unalloyed_quant_cn = alloy_demand_cn1.copy().loc['Unalloyed']
        unalloyed_quant_rw = alloy_demand_rw1.copy().loc['Unalloyed']
        alloy_demand_cn1.drop('Unalloyed',inplace = True)
        alloy_demand_rw1.drop('Unalloyed',inplace = True)

    production = np.array(to_alloy(alloy_demand_cn1.copy(),s2s,prod_spec1.copy()))
    production = np.append(production, np.array(to_alloy(alloy_demand_rw1.copy(),s2s,prod_spec1.copy())))

    for i in prod_spec1.index:
        prod_spec1.loc[i+num_prod,:] = prod_spec1.loc[i,:]
        prod_spec1.loc[i+num_prod,'UNS'] = prod_spec1.loc[i,'UNS'] + '_rw'

    if unalloyed_quant_cn != 0: 
        prod_spec1.loc['Unalloyed CN','High_Cu':'Low_Fe'] = pd.Series([100, 99.8, 0.01, 0, 0.01, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Unalloyed CN','UNS'] = 'Unalloyed CN'
        prod_spec1.loc['Unalloyed CN','Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        prod_spec1.loc['Unalloyed RoW','High_Cu':'Low_Fe'] = pd.Series([100, 99.8, 0.01, 0, 0.01, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Unalloyed RoW','UNS'] = 'Unalloyed RoW'
        prod_spec1.loc['Unalloyed RoW','Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        production = np.append(production, unalloyed_quant_cn)
        production = np.append(production, unalloyed_quant_rw)
        cn_locs = list(np.arange(0,num_prod))
        cn_locs.append(num_prod*2)
        rw_locs = list(np.arange(num_prod,num_prod*2))
        rw_locs.append(num_prod*2+1)

    if fraction_yellows != 0:
        production = np.append(production, sec_ref_prod_cn)
        prod_spec1.loc['Secondary refined CN', 'High_Cu':'Low_Fe'] = pd.Series([100, 50, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Secondary refined CN', 'UNS'] = 'Secondary refined CN'
        prod_spec1.loc['Secondary refined CN', 'Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        production = np.append(production, sec_ref_prod_rw)
        prod_spec1.loc['Secondary refined RoW', 'High_Cu':'Low_Fe'] = pd.Series([100, 50, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Secondary refined RoW', 'UNS'] = 'Secondary refined RoW'
        prod_spec1.loc['Secondary refined RoW', 'Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        cn_locs.append(num_prod*2+2)
        rw_locs.append(num_prod*2+3)
    #         avail['No.1'] = unalloyed_tune*(avail['No.1'] + avail['No.2'])
    #         avail['No.2'] = avail['No.1']

    if type(fruity_alloys) != type(0):
        prod_spec1.loc[:,'Min Recycled Content'] = 0
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1', 'Min Recycled Content'] = fruity_rr[0]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Yellow_Brass', 'Min Recycled Content'] = fruity_rr[1]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Ni_Ag', 'Min Recycled Content'] = fruity_rr[2]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Sn_Bronze', 'Min Recycled Content'] = fruity_rr[3]
        if type(og_rc) != type(0):
            prod_spec1 = prod_spec1.copy()
            prod_spec1.index = prod_spec1.UNS
            if fruity_alloys.index[0] in prod_spec1.index:
                prod_spec1.drop(fruity_alloys.index, inplace=True)
            prod_spec1.loc[:,'Min Recycled Content'] = og_rc
            
        for i in fruity_alloys1.index:
            prod_spec1.loc[i,:] = fruity_alloys1.loc[i,:]
        production = np.append(production, fruity_alloys1.loc[:,'Quantity'].values)
        cn_locs.extend(list(np.arange(num_prod*2+4, num_prod*2+4+fruity_alloys1.shape[0])))

    # Product specs
    element, max_spec, min_spec = multidict({
        "Cu":[np.array(prod_spec1.High_Cu.values),np.array(prod_spec1.Low_Cu.values)],
        "Zn":[np.array(prod_spec1.High_Zn.values),np.array(prod_spec1.Low_Zn.values)],
        "Pb":[np.array(prod_spec1.High_Pb.values),np.array(prod_spec1.Low_Pb.values)],
        "Sn":[np.array(prod_spec1.High_Sn.values),np.array(prod_spec1.Low_Sn.values)],
        "Ni":[np.array(prod_spec1.High_Ni.values),np.array(prod_spec1.Low_Ni.values)],
        "Al":[np.array(prod_spec1.High_Al.values),np.array(prod_spec1.Low_Al.values)],
        "Mn":[np.array(prod_spec1.High_Mn.values),np.array(prod_spec1.Low_Mn.values)],
        "Fe":[np.array(prod_spec1.High_Fe.values),np.array(prod_spec1.Low_Fe.values)]       
    })

    alloys = np.array(prod_spec1.UNS.values)
    scraps = sorted(prod_spec1.loc[:,'Alloy Type'].unique())
    if type(fruity_alloys) != type(0):
        min_recycled_content = np.array(prod_spec1.loc[:,'Min Recycled Content'])

    num_alloys = len(alloys)
    alloy_blend = alloys[0:num_alloys]
    
    confidence = 0.95
    s = confidence*2 - 1
    CC = True

    m = Model('Blending')

    raw_demand = m.addVars(alloy_blend, raw, name='raw_demand')

    for a_i in range(0,num_alloys):
        a = alloy_blend[a_i]
        # Specs constraints: if CC, use chance-constrained; else use deterministic
        if CC:
            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 + (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) <= max_spec['Cu'][a_i] * production[a_i], "spec_Cu_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 - (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) >= min_spec['Cu'][a_i] * production[a_i], "spec_Cu_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 + (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) <= max_spec['Zn'][a_i] * production[a_i], "spec_Zn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 - (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) >= min_spec['Zn'][a_i] * production[a_i], "spec_Zn_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 + (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) <= max_spec['Pb'][a_i] * production[a_i], "spec_Pb_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 - (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) >= min_spec['Pb'][a_i] * production[a_i], "spec_Pb_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 + (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) <= max_spec['Sn'][a_i] * production[a_i], "spec_Sn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 - (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) >= min_spec['Sn'][a_i] * production[a_i], "spec_Sn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 + (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) <= max_spec['Ni'][a_i] * production[a_i], "spec_Ni_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 - (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) >= min_spec['Ni'][a_i] * production[a_i], "spec_Ni_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 + (High_Al[i] - Low_Al[i])/2*s) for i in raw) <= max_spec['Al'][a_i] * production[a_i], "spec_Al_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 - (High_Al[i] - Low_Al[i])/2*s) for i in raw) >= min_spec['Al'][a_i] * production[a_i], "spec_Al_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 + (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) <= max_spec['Mn'][a_i] * production[a_i], "spec_Mn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 - (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) >= min_spec['Mn'][a_i] * production[a_i], "spec_Mn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 + (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) <= max_spec['Fe'][a_i] * production[a_i], "spec_Fe_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 - (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) >= min_spec['Fe'][a_i] * production[a_i], "spec_Fe_lo")        
        else: 
            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Cu[i] + Low_Cu[i])/2 for i in raw), min_spec['Cu'][a_i] * production[a_i], max_spec['Cu'][a_i] * production[a_i], "spec_Cu")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Zn[i] + Low_Zn[i])/2 for i in raw), min_spec['Zn'][a_i] * production[a_i], max_spec['Zn'][a_i] * production[a_i], "spec_Zn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Pb[i] + Low_Pb[i])/2 for i in raw), min_spec['Pb'][a_i] * production[a_i], max_spec['Pb'][a_i] * production[a_i], "spec_Pb")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Sn[i] + Low_Sn[i])/2 for i in raw), min_spec['Sn'][a_i] * production[a_i], max_spec['Sn'][a_i] * production[a_i], "spec_Sn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Ni[i] + Low_Ni[i])/2 for i in raw), min_spec['Ni'][a_i] * production[a_i], max_spec['Ni'][a_i] * production[a_i], "spec_Ni")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Al[i] + Low_Al[i])/2 for i in raw), min_spec['Al'][a_i] * production[a_i], max_spec['Al'][a_i] * production[a_i], "spec_Al")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Mn[i] + Low_Mn[i])/2 for i in raw), min_spec['Mn'][a_i] * production[a_i], max_spec['Mn'][a_i] * production[a_i], "spec_Mn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Fe[i] + Low_Fe[i])/2 for i in raw), min_spec['Fe'][a_i] * production[a_i], max_spec['Fe'][a_i] * production[a_i], "spec_Fe")

        # Mass constraint
        m.addConstr(quicksum(raw_demand[(a,i)] for i in raw) >= production[a_i], "mass")
        for i in raw:
            m.addConstr(raw_demand[(a,i)] >= 0, 'Just dont be negative')
        if type(fruity_alloys) != type(0):
            if rc_variance != 0:
                if fruity_rr[0] != 0 and a_i >= num_prod*2+4:
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) <= production[a_i]*(1-min_recycled_content[a_i]), 'recycled content')
        #             print('A:',a)
                elif fruity_rr[0] != 0 or a_i < num_prod*2+4:
        #             print('B:',a)
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) <= production[a_i]*(1-min_recycled_content[a_i]*rc_variance), 'recycled content')
                else:
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) == production[a_i], 'no recycled content')
        #             print(a, 'no recycled content')

            else:
                if fruity_rr[0] != 0 and a_i >= num_prod*2+4:
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) <= production[a_i]*(1-min_recycled_content[a_i]), 'recycled content')
                elif fruity_rr[0] != 0 or a_i < num_prod*2+4:
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) <= production[a_i], 'recycled content')
                else:
                    m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) == production[a_i], 'no recycled content')
        #             print(a, 'no recycled content')
        
        # No No.2
        if 'Secondary refined' in a:
            for rs in refs + list(['No.1']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no #1 or refined scrap")
            for rs in list(['Yellow_Brass', 'Pb_Yellow_Brass', 'Cartridge']):
                if a_i in cn_locs:
                    m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_cn[rs], "Low yellows")
                elif a_i in rw_locs:
                    m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_rw[rs], "Low yellows")
            if a_i in cn_locs:
                m.addConstr(raw_demand[(a,rs)] <= fraction_yellows*2*avail_cn[rs], 'Low Sn Bronze')
            elif a_i in rw_locs:
                m.addConstr(raw_demand[(a,rs)] <= fraction_yellows*2*avail_rw[rs], 'Low Sn Bronze')
            if pir_pcr != 0:
                for rs in list(new_scrap_alloys_cn1.index):
                    if a_i in cn_locs:
                        m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_cn[rs], "limit new scrap")
                    elif a_i in rw_locs:
                        m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_rw[rs], "limit new scrap")
        elif 'Unalloyed' in a:
            for rs in refs[1:] + list(['No.2']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no refined aside from Cu")
        else:
            m.addConstr(raw_demand[(a,'No.2')] == 0, "no #2")
            
        if pir_fraction != -1 and a in fruity_alloys.index:
            if pir_fraction == 0:
                m.addConstr(quicksum(raw_demand[a,i] for i in new_scrap_alloys_cn1.index) == 0)
            elif pir_fraction == 1:
                m.addConstr(quicksum(raw_demand[a,i] for i in new_scrap_alloys_cn1.index) == 1)
            else:
                m.addConstr(quicksum(raw_demand[a,i] for i in new_scrap_alloys_cn1.index) <= production[a_i]*(pir_fraction+wiggle))
                m.addConstr(quicksum(raw_demand[a,i] for i in new_scrap_alloys_cn1.index) >= production[a_i]*(pir_fraction-wiggle))

    new_quant_cn = {}
    new_quant_rw = {}
    new_cost_cn = {}
    new_cost_rw = {}
    n = 1000
    slope = -1
    width_cn = sum(production[cn_locs])/n
    width_rw = sum(production[rw_locs])/n
    raw_demand_tot = m.addVars(raw, name='raw_demand_tot')
    raw_demand_tot_cn = m.addVars(raw, name = 'raw_demand_tot_cn')
    raw_demand_tot_rw = m.addVars(raw, name = 'raw_demand_tot_rw')
    #     if fraction_yellows != 0:
    #         m.addConstr(raw_demand_tot['No.1'] + raw_demand_tot['No.2'] <= avail['No.1'], 'unalloyed constraint') # since avail['No.1'] == avail['No.2'] == total unalloyed avail
    m.addConstr(raw_demand_tot['Ref_Cu'] == ref_demand_cn+ref_demand_rw, 'Ref_Cu == demand')
    for scrap in raw:
        m.addConstr(quicksum(raw_demand[(a,scrap)] for a in alloy_blend[cn_locs]) == raw_demand_tot_cn[scrap])
        m.addConstr(quicksum(raw_demand[(a,scrap)] for a in alloy_blend[rw_locs]) == raw_demand_tot_rw[scrap])

        new_quant_cn.update({scrap: np.linspace(0,sum(production[cn_locs]),n)})
        new_quant_rw.update({scrap: np.linspace(0,sum(production[rw_locs]),n)})
        if scrap in scraps:
    #             if type(prior_demand) == type(0):
            new_cost_cn.update({scrap: np.cumsum(width_cn*avg_cost_curve(new_quant_cn[scrap],cost_cn[scrap],avail_cn[scrap],slope))}) # effectively integrating under the curve
            new_cost_rw.update({scrap: np.cumsum(width_rw*avg_cost_curve(new_quant_rw[scrap],cost_rw[scrap],avail_rw[scrap],slope))}) # effectively integrating under the curve
    #             else:
    #                 adder = 0.2*abs(prior_demand[scrap]-new_quant[scrap])*cost[scrap]
    #                 adder[((new_quant[scrap]<prior_demand[scrap]).sum()):] = 0
    #                 new_cost.update({scrap: np.cumsum(width*avg_cost_curve(new_quant[scrap],cost[scrap],avail[scrap],slope))+adder})
            m.addConstr(raw_demand_tot_cn[scrap] <= avail_cn[scrap])
            m.addConstr(raw_demand_tot_rw[scrap] <= avail_rw[scrap])
        elif scrap == 'Ref_Cu' and weight_cu == 1:
            new_cost_cn.update({scrap: multiplier*cost_cn['Ref_Cu']*abs((ref_demand_cn - new_quant_cn['Ref_Cu'])*0.95)**2/(ref_demand_cn)\
                             + cost_cn['Ref_Cu']*new_quant_cn['Ref_Cu']})
            new_cost_rw.update({scrap: multiplier*cost_rw['Ref_Cu']*abs((ref_demand_rw - new_quant_rw['Ref_Cu'])*0.95)**2/(ref_demand_rw)\
                             + cost_rw['Ref_Cu']*new_quant_rw['Ref_Cu']})
    #             new_cost.update({scrap: cost['Ref_Cu']*abs((ref_demand - new_quant['Ref_Cu'])*multiplier)**2/(ref_demand)})        
        elif not np.isnan(avail_cn[scrap]) and pir_pcr == 1:
            m.addConstr(raw_demand_tot_cn[scrap] <= avail_cn[scrap])
            m.addConstr(raw_demand_tot_rw[scrap] <= avail_rw[scrap])
            new_cost_cn.update({scrap: cost_cn[scrap]*new_quant_cn[scrap]})
            new_cost_rw.update({scrap: cost_rw[scrap]*new_quant_rw[scrap]})
        else:
            new_cost_cn.update({scrap: cost_cn[scrap]*new_quant_cn[scrap]})
            new_cost_rw.update({scrap: cost_rw[scrap]*new_quant_rw[scrap]})
        m.addConstr(raw_demand_tot[scrap] == raw_demand_tot_cn[scrap] + raw_demand_tot_rw[scrap])
        m.setPWLObj(raw_demand_tot_cn[scrap],new_quant_cn[scrap],new_cost_cn[scrap])
        m.setPWLObj(raw_demand_tot_rw[scrap],new_quant_rw[scrap],new_cost_rw[scrap])

    # Optimize!
    m.update()
    m.setParam( 'OutputFlag', False )
    m.optimize()

    # Return results
    demand = pd.Series(m.getAttr('x', raw_demand_tot))
    demand_cn = pd.Series(m.getAttr('x', raw_demand_tot_cn))
    demand_rw = pd.Series(m.getAttr('x', raw_demand_tot_rw))

    if fraction_yellows != 0:
        idx = pd.IndexSlice
        refined_secondary_demand_cn = pd.Series(m.getAttr('x', raw_demand)).loc[idx['Secondary refined CN',:]]
        refined_secondary_demand_rw = pd.Series(m.getAttr('x', raw_demand)).loc[idx['Secondary refined RoW',:]]
        actual_rc = (pd.Series(m.getAttr('x', raw_demand)).groupby(level=0).sum() - pd.Series(m.getAttr('x', raw_demand)).loc[idx[:,refs]].groupby(level=0).sum())/pd.Series(m.getAttr('x', raw_demand)).groupby(level=0).sum()
        actual_rcv = (pd.Series(m.getAttr('x', raw_demand)).groupby(level=0).sum() - pd.Series(m.getAttr('x', raw_demand)).loc[idx[:,refs]].groupby(level=0).sum())
        if type(fruity_alloys) != type(0):
            fruity_demand = pd.DataFrame(0, index = demand.index, columns = fruity_alloys.index)
            for i in fruity_alloys.index:
                fruity_demand.loc[:, i] = pd.Series(m.getAttr('x', raw_demand)).loc[idx[i,:]]
#             print('No.1: PIR Quant:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'No.1'].sum(), 'PIR Fraction:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'No.1'].sum()/fruity_demand.loc[:,fruity_alloys1.loc[:,'Alloy Type'] == 'No.1'].sum())
#             print('YeBr: PIR Quant:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Yellow_Brass'].sum(), 'PIR Fraction:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Yellow_Brass'].sum()/fruity_demand.loc[:,fruity_alloys1.loc[:,'Alloy Type'] == 'Yellow_Brass'].sum())
#             print('NiAg: PIR Quant:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Ni_Ag'].sum(), 'PIR Fraction:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Ni_Ag'].sum()/fruity_demand.loc[:,fruity_alloys1.loc[:,'Alloy Type'] == 'Ni_Ag'].sum())
#             print('SnBr: PIR Quant:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Sn_Bronze'].sum(), 'PIR Fraction:\t', fruity_demand.loc[new_scrap_alloys_cn1.index,fruity_alloys1.loc[:,'Alloy Type'] == 'Sn_Bronze'].sum()/fruity_demand.loc[:,fruity_alloys1.loc[:,'Alloy Type'] == 'Sn_Bronze'].sum())
            
            return demand_cn, demand_rw, refined_secondary_demand_cn, refined_secondary_demand_rw, fruity_demand.T.values, actual_rc, actual_rcv
        else:
            return demand_cn, demand_rw, refined_secondary_demand_cn, refined_secondary_demand_rw
    else:
        return demand_cn, demand_rw
        # print(n)

def blend_ff_old(availability_cn, availability_rw, alloy_demand_cn, alloy_demand_rw, raw_price_cn, raw_price_rw, s2s, 
             prod_spec, raw_spec, sec_ref_prod_cn = 0, sec_ref_prod_rw = 0, ref_demand_cn = 0, ref_demand_rw = 0, 
             fraction_yellows = 0, unalloyed_tune = 1, fruity_alloys=0, fruity_rr = [0,0,0,0], pir_pcr = 0, 
             pir_fraction = -1, new_scrap_alloys_cn = 0, new_scrap_alloys_rw = 0, pir_price = 1.09, cn_scrap_imports = 0):
    '''Outputs demand for each scrap grade given demand for unalloyed and alloyed shapes for semis. With sec_ref_prod 
    specified, includes secondary refined production in the blending optimization, where secondary refineries can consume all
    scrap types except No.1, although yellow brasses are limited in refined use (based on USGS [values] and SMM [checkmarks])
    using the fraction_yellows variable. Giving ref_demand causes weighting within the optimization to put Ref_Cu consumption 
    closer to that value, where both No.1 and No.2 availabilities were set to the total unalloyed value and unalloyed_tune
    multiplies by that value to change unalloyed scrap use. '''
    
#     pir_pcr = 0 # zero excludes pir consideration, 1 allows new scrap from that year to be used
#     pir_fraction = -1 # set to zero to keep fruity from using PIR, -1 for no constraints on fruity PIR content
    weight_cu = 0
    multiplier = 1.5
#     pir_price = 1#1.09

    scraps = list(['Al_Bronze', 'Cartridge', 'Mn_Bronze', 'Ni_Ag', 'No.1', 'No.2', 'Ocean', 'Pb_Sn_Bronze', 'Pb_Yellow_Brass', 'Red_Brass', 'Sn_Bronze', 'Yellow_Brass'])
    refs = list(['Ref_Cu', 'Ref_Al', 'Ref_Fe', 'Ref_Mn', 'Ref_Ni', 'Ref_Pb', 'Ref_Sn', 'Ref_Zn'])

    prod_spec1 = prod_spec.copy()
    num_prod = prod_spec1.shape[0]
    alloy_demand_cn1 = alloy_demand_cn.copy()
    alloy_demand_rw1 = alloy_demand_rw.copy()
    new_scrap_alloys_cn1 = new_scrap_alloys_cn.copy()
    new_scrap_alloys_rw1 = new_scrap_alloys_rw.copy()
    if type(fruity_alloys) != type(0):
        fruity_alloys1 = fruity_alloys.copy()
    raw_spec_cn1 = raw_spec.copy()
    raw_spec_rw1 = raw_spec.copy()
    raw_spec_cn1['Price'] = raw_price_cn
    raw_spec_rw1['Price'] = raw_price_rw
    if type(cn_scrap_imports) != type(0):
        availabliity_cn -= cn_scrap_imports.loc[:,'Availability']
    raw_spec_cn1['Availability'] = availability_cn
    raw_spec_rw1['Availability'] = availability_rw
    if pir_pcr == 1:
#         for sc in scraps: # this is from a previous attempt where each new scrap's price was determined relative to its corresponding scrap type - it appears that historical consistency aligns with it corresponding with No.1 instead
#             new_scrap_alloys_cn1.loc[new_scrap_alloys_cn1.loc[:,'Alloy Type']==sc,'Price'] = raw_price_cn.loc[sc] * pir_price
#             new_scrap_alloys_rw1.loc[new_scrap_alloys_rw1.loc[:,'Alloy Type']==sc,'Price'] = raw_price_rw.loc[sc] * pir_price
#         new_scrap_alloys_cn1.drop(inplace=True, columns=['Alloy Type'])
#         new_scrap_alloys_rw1.drop(inplace=True, columns=['Alloy Type']) # had also required changing the new_scrap_alloys_rw_year_i = new_scrap_alloys_compositions.copy().loc[:,'High_Cu':] from 'Ref_Cu': to 'Alloy Type':
        new_scrap_alloys_cn1.loc[:, 'Price'] = raw_price_cn.loc['No.1'] * pir_price
        new_scrap_alloys_rw1.loc[:, 'Price'] = raw_price_rw.loc['No.1'] * pir_price
        raw_spec_cn1 = pd.concat([raw_spec_cn1,new_scrap_alloys_cn1], sort=False)
        raw_spec_rw1 = pd.concat([raw_spec_rw1,new_scrap_alloys_rw1], sort=False)
    if type(cn_scrap_imports) != type(0):
        cn_scrap_imports['Price'] = raw_price_cn
        rw_scrap_imports = cn_scrap_imports.copy()
        rw_scrap_imports.loc[:,'Availability'] = 0
        raw_spec_cn1 = pd.concat([raw_spec_cn1, cn_scrap_imports], sort=False)
        raw_spec_rw1 = pd.concat([raw_spec_cn1, rw_scrap_imports], sort=False)

    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn, High_Ni, Low_Ni,\
        High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost_cn, avail_cn = multidict(raw_spec_cn1.T.to_dict('list'))
    raw, High_Cu, Low_Cu, High_Zn, Low_Zn, High_Pb, Low_Pb, High_Sn, Low_Sn, High_Ni, Low_Ni,\
        High_Al, Low_Al, High_Mn, Low_Mn, High_Fe, Low_Fe, cost_rw, avail_rw = multidict(raw_spec_rw1.T.to_dict('list'))
    
    if type(fruity_alloys) != type(0):
        if 'Unalloyed' in alloy_demand_cn1.index:
            alloy_demand_cn1.loc['Unalloyed'] -= fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1','Quantity'].sum()
        alloy_demand_cn1.loc['Electronic'] -= fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] != 'No.1','Quantity'].sum()

    unalloyed_quant_cn = 0
    unalloyed_quant_rw = 0
    if 'Unalloyed' in alloy_demand_cn1.index:
        unalloyed_quant_cn = alloy_demand_cn1.copy().loc['Unalloyed']
        unalloyed_quant_rw = alloy_demand_rw1.copy().loc['Unalloyed']
        alloy_demand_cn1.drop('Unalloyed',inplace = True)
        alloy_demand_rw1.drop('Unalloyed',inplace = True)

    production = np.array(to_alloy(alloy_demand_cn1.copy(),s2s,prod_spec1.copy()))
    production = np.append(production, np.array(to_alloy(alloy_demand_rw1.copy(),s2s,prod_spec1.copy())))

    for i in prod_spec1.index:
        prod_spec1.loc[i+num_prod,:] = prod_spec1.loc[i,:]
        prod_spec1.loc[i+num_prod,'UNS'] = prod_spec1.loc[i,'UNS'] + '_rw'

    if unalloyed_quant_cn != 0: 
        prod_spec1.loc['Unalloyed CN','High_Cu':'Low_Fe'] = pd.Series([100, 99.8, 0.01, 0, 0.01, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Unalloyed CN','UNS'] = 'Unalloyed CN'
        prod_spec1.loc['Unalloyed CN','Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        prod_spec1.loc['Unalloyed RoW','High_Cu':'Low_Fe'] = pd.Series([100, 99.8, 0.01, 0, 0.01, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Unalloyed RoW','UNS'] = 'Unalloyed RoW'
        prod_spec1.loc['Unalloyed RoW','Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        production = np.append(production, unalloyed_quant_cn)
        production = np.append(production, unalloyed_quant_rw)
        cn_locs = list(np.arange(0,num_prod))
        cn_locs.append(num_prod*2)
        rw_locs = list(np.arange(num_prod,num_prod*2))
        rw_locs.append(num_prod*2+1)

    if fraction_yellows != 0:
        production = np.append(production, sec_ref_prod_cn)
        prod_spec1.loc['Secondary refined CN', 'High_Cu':'Low_Fe'] = pd.Series([100, 50, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Secondary refined CN', 'UNS'] = 'Secondary refined CN'
        prod_spec1.loc['Secondary refined CN', 'Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        production = np.append(production, sec_ref_prod_rw)
        prod_spec1.loc['Secondary refined RoW', 'High_Cu':'Low_Fe'] = pd.Series([100, 50, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0], index = prod_spec1.loc[:,'High_Cu':'Low_Fe'].columns)
        prod_spec1.loc['Secondary refined RoW', 'UNS'] = 'Secondary refined RoW'
        prod_spec1.loc['Secondary refined RoW', 'Alloy Type':'Category'] = prod_spec1.iloc[0,:].loc['Alloy Type':'Category']
        cn_locs.append(num_prod*2+2)
        rw_locs.append(num_prod*2+3)
    #         avail['No.1'] = unalloyed_tune*(avail['No.1'] + avail['No.2'])
    #         avail['No.2'] = avail['No.1']

    if type(fruity_alloys) != type(0):
        prod_spec1.loc[:,'Min Recycled Content'] = 0
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'No.1', 'Min Recycled Content'] = fruity_rr[0]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Yellow_Brass', 'Min Recycled Content'] = fruity_rr[1]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Ni_Ag', 'Min Recycled Content'] = fruity_rr[2]
        fruity_alloys1.loc[fruity_alloys1.loc[:,'Alloy Type'] == 'Sn_Bronze', 'Min Recycled Content'] = fruity_rr[3]
        for i in fruity_alloys1.index:
            prod_spec1.loc[i,:] = fruity_alloys1.loc[i,:]
        production = np.append(production, fruity_alloys1.loc[:,'Quantity'].values)
        cn_locs.extend(list(np.arange(num_prod*2+4, num_prod*2+4+fruity_alloys.shape[0])))

    # Product specs
    element, max_spec, min_spec = multidict({
        "Cu":[np.array(prod_spec1.High_Cu.values),np.array(prod_spec1.Low_Cu.values)],
        "Zn":[np.array(prod_spec1.High_Zn.values),np.array(prod_spec1.Low_Zn.values)],
        "Pb":[np.array(prod_spec1.High_Pb.values),np.array(prod_spec1.Low_Pb.values)],
        "Sn":[np.array(prod_spec1.High_Sn.values),np.array(prod_spec1.Low_Sn.values)],
        "Ni":[np.array(prod_spec1.High_Ni.values),np.array(prod_spec1.Low_Ni.values)],
        "Al":[np.array(prod_spec1.High_Al.values),np.array(prod_spec1.Low_Al.values)],
        "Mn":[np.array(prod_spec1.High_Mn.values),np.array(prod_spec1.Low_Mn.values)],
        "Fe":[np.array(prod_spec1.High_Fe.values),np.array(prod_spec1.Low_Fe.values)]       
    })

    alloys = np.array(prod_spec1.UNS.values)
    scraps = sorted(prod_spec1.loc[:,'Alloy Type'].unique())
    if type(fruity_alloys) != type(0):
        min_recycled_content = np.array(prod_spec1.loc[:,'Min Recycled Content'])

    num_alloys = len(alloys)
    alloy_blend = alloys[0:num_alloys]
    
    confidence = 0.95
    s = confidence*2 - 1
    CC = True

    m = Model('Blending')

    raw_demand = m.addVars(alloy_blend, raw, name='raw_demand')

    for a_i in range(0,num_alloys):
        a = alloy_blend[a_i]
        # Specs constraints: if CC, use chance-constrained; else use deterministic
        if CC:
            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 + (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) <= max_spec['Cu'][a_i] * production[a_i], "spec_Cu_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Cu[i] + Low_Cu[i])/2 - (High_Cu[i] - Low_Cu[i])/2*s) for i in raw) >= min_spec['Cu'][a_i] * production[a_i], "spec_Cu_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 + (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) <= max_spec['Zn'][a_i] * production[a_i], "spec_Zn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Zn[i] + Low_Zn[i])/2 - (High_Zn[i] - Low_Zn[i])/2*s) for i in raw) >= min_spec['Zn'][a_i] * production[a_i], "spec_Zn_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 + (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) <= max_spec['Pb'][a_i] * production[a_i], "spec_Pb_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Pb[i] + Low_Pb[i])/2 - (High_Pb[i] - Low_Pb[i])/2*s) for i in raw) >= min_spec['Pb'][a_i] * production[a_i], "spec_Pb_lo")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 + (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) <= max_spec['Sn'][a_i] * production[a_i], "spec_Sn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Sn[i] + Low_Sn[i])/2 - (High_Sn[i] - Low_Sn[i])/2*s) for i in raw) >= min_spec['Sn'][a_i] * production[a_i], "spec_Sn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 + (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) <= max_spec['Ni'][a_i] * production[a_i], "spec_Ni_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Ni[i] + Low_Ni[i])/2 - (High_Ni[i] - Low_Ni[i])/2*s) for i in raw) >= min_spec['Ni'][a_i] * production[a_i], "spec_Ni_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 + (High_Al[i] - Low_Al[i])/2*s) for i in raw) <= max_spec['Al'][a_i] * production[a_i], "spec_Al_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Al[i] + Low_Al[i])/2 - (High_Al[i] - Low_Al[i])/2*s) for i in raw) >= min_spec['Al'][a_i] * production[a_i], "spec_Al_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 + (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) <= max_spec['Mn'][a_i] * production[a_i], "spec_Mn_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Mn[i] + Low_Mn[i])/2 - (High_Mn[i] - Low_Mn[i])/2*s) for i in raw) >= min_spec['Mn'][a_i] * production[a_i], "spec_Mn_lo")        

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 + (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) <= max_spec['Fe'][a_i] * production[a_i], "spec_Fe_up")

            m.addConstr(
                    quicksum(raw_demand[(a,i)] * ((High_Fe[i] + Low_Fe[i])/2 - (High_Fe[i] - Low_Fe[i])/2*s) for i in raw) >= min_spec['Fe'][a_i] * production[a_i], "spec_Fe_lo")        
        else: 
            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Cu[i] + Low_Cu[i])/2 for i in raw), min_spec['Cu'][a_i] * production[a_i], max_spec['Cu'][a_i] * production[a_i], "spec_Cu")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Zn[i] + Low_Zn[i])/2 for i in raw), min_spec['Zn'][a_i] * production[a_i], max_spec['Zn'][a_i] * production[a_i], "spec_Zn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Pb[i] + Low_Pb[i])/2 for i in raw), min_spec['Pb'][a_i] * production[a_i], max_spec['Pb'][a_i] * production[a_i], "spec_Pb")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Sn[i] + Low_Sn[i])/2 for i in raw), min_spec['Sn'][a_i] * production[a_i], max_spec['Sn'][a_i] * production[a_i], "spec_Sn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Ni[i] + Low_Ni[i])/2 for i in raw), min_spec['Ni'][a_i] * production[a_i], max_spec['Ni'][a_i] * production[a_i], "spec_Ni")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Al[i] + Low_Al[i])/2 for i in raw), min_spec['Al'][a_i] * production[a_i], max_spec['Al'][a_i] * production[a_i], "spec_Al")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Mn[i] + Low_Mn[i])/2 for i in raw), min_spec['Mn'][a_i] * production[a_i], max_spec['Mn'][a_i] * production[a_i], "spec_Mn")

            m.addRange(
                    quicksum(raw_demand[(a,i)] * (High_Fe[i] + Low_Fe[i])/2 for i in raw), min_spec['Fe'][a_i] * production[a_i], max_spec['Fe'][a_i] * production[a_i], "spec_Fe")

        # Mass constraint
        m.addConstr(quicksum(raw_demand[(a,i)] for i in raw) >= production[a_i], "mass")
        if type(fruity_alloys) != type(0):
            if fruity_rr[0] != 0 or a_i < num_prod*2+4:
                m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) <= production[a_i]*(1-min_recycled_content[a_i]), 'recycled content')
            else:
                m.addConstr(quicksum(raw_demand[(a,i)] for i in refs) == production[a_i], 'no recycled content')
    #             print(a, 'no recycled content')

        # No No.2
        if 'Secondary refined' in a:
            for rs in refs + list(['No.1']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no #1 or refined scrap")
            for rs in list(['Yellow_Brass', 'Pb_Yellow_Brass', 'Cartridge']):
                if a_i in cn_locs:
                    m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_cn[rs], "Low yellows")
                elif a_i in rw_locs:
                    m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_rw[rs], "Low yellows")
            if a_i in cn_locs:
                m.addConstr(raw_demand[(a,rs)] <= fraction_yellows*2*avail_cn[rs], 'Low Sn Bronze')
            elif a_i in rw_locs:
                m.addConstr(raw_demand[(a,rs)] <= fraction_yellows*2*avail_rw[rs], 'Low Sn Bronze')
            if pir_pcr != 0:
                for rs in list(new_scrap_alloys_cn1.index):
                    if a_i in cn_locs:
                        m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_cn[rs], "limit new scrap")
                    elif a_i in rw_locs:
                        m.addConstr(raw_demand[(a, rs)] <= fraction_yellows*avail_rw[rs], "limit new scrap")
        elif 'Unalloyed' in a:
            for rs in refs[1:] + list(['No.2']):
                m.addConstr(raw_demand[(a, rs)] == 0, "no refined aside from Cu")
        else:
            m.addConstr(raw_demand[(a,'No.2')] == 0, "no #2")
        if pir_fraction != -1 and a in fruity_alloys.index:
            m.addConstr(quicksum(raw_demand[a,i] for i in new_scrap_alloys_cn1.index) <= production[a_i]*pir_fraction)

    new_quant_cn = {}
    new_quant_rw = {}
    new_cost_cn = {}
    new_cost_rw = {}
    n = 1000
    slope = -1
    width_cn = sum(production[cn_locs])/n
    width_rw = sum(production[rw_locs])/n
    raw_demand_tot = m.addVars(raw, name='raw_demand_tot')
    raw_demand_tot_cn = m.addVars(raw, name = 'raw_demand_tot_cn')
    raw_demand_tot_rw = m.addVars(raw, name = 'raw_demand_tot_rw')
    #     if fraction_yellows != 0:
    #         m.addConstr(raw_demand_tot['No.1'] + raw_demand_tot['No.2'] <= avail['No.1'], 'unalloyed constraint') # since avail['No.1'] == avail['No.2'] == total unalloyed avail
    m.addConstr(raw_demand_tot['Ref_Cu'] == ref_demand_cn+ref_demand_rw, 'Ref_Cu == demand')
    for scrap in raw:
        m.addConstr(quicksum(raw_demand[(a,scrap)] for a in alloy_blend[cn_locs]) == raw_demand_tot_cn[scrap])
        m.addConstr(quicksum(raw_demand[(a,scrap)] for a in alloy_blend[rw_locs]) == raw_demand_tot_rw[scrap])

        new_quant_cn.update({scrap: np.linspace(0,sum(production[cn_locs]),n)})
        new_quant_rw.update({scrap: np.linspace(0,sum(production[rw_locs]),n)})
        if scrap in scraps:
    #             if type(prior_demand) == type(0):
            new_cost_cn.update({scrap: np.cumsum(width_cn*avg_cost_curve(new_quant_cn[scrap],cost_cn[scrap],avail_cn[scrap],slope))}) # effectively integrating under the curve
            new_cost_rw.update({scrap: np.cumsum(width_rw*avg_cost_curve(new_quant_rw[scrap],cost_rw[scrap],avail_rw[scrap],slope))}) # effectively integrating under the curve
    #             else:
    #                 adder = 0.2*abs(prior_demand[scrap]-new_quant[scrap])*cost[scrap]
    #                 adder[((new_quant[scrap]<prior_demand[scrap]).sum()):] = 0
    #                 new_cost.update({scrap: np.cumsum(width*avg_cost_curve(new_quant[scrap],cost[scrap],avail[scrap],slope))+adder})
            m.addConstr(raw_demand_tot_cn[scrap] <= avail_cn[scrap])
            m.addConstr(raw_demand_tot_rw[scrap] <= avail_rw[scrap])
        elif scrap == 'Ref_Cu' and weight_cu == 1:
            new_cost_cn.update({scrap: multiplier*cost_cn['Ref_Cu']*abs((ref_demand_cn - new_quant_cn['Ref_Cu'])*0.95)**2/(ref_demand_cn)\
                             + cost_cn['Ref_Cu']*new_quant_cn['Ref_Cu']})
            new_cost_rw.update({scrap: multiplier*cost_rw['Ref_Cu']*abs((ref_demand_rw - new_quant_rw['Ref_Cu'])*0.95)**2/(ref_demand_rw)\
                             + cost_rw['Ref_Cu']*new_quant_rw['Ref_Cu']})
    #             new_cost.update({scrap: cost['Ref_Cu']*abs((ref_demand - new_quant['Ref_Cu'])*multiplier)**2/(ref_demand)})        
        elif not np.isnan(avail_cn[scrap]) and pir_pcr == 1:
            m.addConstr(raw_demand_tot_cn[scrap] <= avail_cn[scrap])
            m.addConstr(raw_demand_tot_rw[scrap] <= avail_rw[scrap])
            new_cost_cn.update({scrap: cost_cn[scrap]*new_quant_cn[scrap]})
            new_cost_rw.update({scrap: cost_rw[scrap]*new_quant_rw[scrap]})
        else:
            new_cost_cn.update({scrap: cost_cn[scrap]*new_quant_cn[scrap]})
            new_cost_rw.update({scrap: cost_rw[scrap]*new_quant_rw[scrap]})
        m.addConstr(raw_demand_tot[scrap] == raw_demand_tot_cn[scrap] + raw_demand_tot_rw[scrap])
        m.setPWLObj(raw_demand_tot_cn[scrap],new_quant_cn[scrap],new_cost_cn[scrap])
        m.setPWLObj(raw_demand_tot_rw[scrap],new_quant_rw[scrap],new_cost_rw[scrap])

    # Optimize!
    m.update()
    m.setParam( 'OutputFlag', False )
    m.optimize()

    # Return results
    demand = pd.Series(m.getAttr('x', raw_demand_tot))
    demand_cn = pd.Series(m.getAttr('x', raw_demand_tot_cn))
    demand_rw = pd.Series(m.getAttr('x', raw_demand_tot_rw))

    if fraction_yellows != 0:
        idx = pd.IndexSlice
        refined_secondary_demand_cn = pd.Series(m.getAttr('x', raw_demand)).loc[idx['Secondary refined CN',:]]
        refined_secondary_demand_rw = pd.Series(m.getAttr('x', raw_demand)).loc[idx['Secondary refined RoW',:]]
        if type(fruity_alloys) != type(0):
            fruity_demand = pd.DataFrame(0, index = demand.index, columns = fruity_alloys.index)
            for i in fruity_alloys.index:
                fruity_demand.loc[:, i] = pd.Series(m.getAttr('x', raw_demand)).loc[idx[i,:]]
            return demand_cn, demand_rw, refined_secondary_demand_cn, refined_secondary_demand_rw, fruity_demand.sum(axis=1)
        else:
            return demand_cn, demand_rw, refined_secondary_demand_cn, refined_secondary_demand_rw
    else:
        return demand_cn, demand_rw
        # print(n)
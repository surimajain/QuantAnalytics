from datetime import datetime
from datetime import timedelta
import QuantLib as quant
import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import os, sys
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
# import functions.utils.GSODataFetcher as gsodatafetcher

def getCalendar(CalendarDF):
    isHoliDay = CalendarDF['BusinessDateIndicator'] == 'N'

    CalendarDF = CalendarDF[isHoliDay]

    myCalendar = quant.WeekendsOnly()
    CalendarDF['GregorianDate'] = pd.to_datetime(CalendarDF['GregorianDate'],
                                                         format='%Y-%m-%dT%H:%M:%S.000')
    dates = CalendarDF['GregorianDate'].values
    for date in dates:
        quantDate = quant.Date(pd.to_datetime(date).day, pd.to_datetime(date).month, pd.to_datetime(date).year)
        myCalendar.addHoliday(quantDate)
    return myCalendar

dayCountsDataset={
    "ACT/ACT":quant.ActualActual(),
    "ACT/360":quant.Actual360(),
    "ACT/365F":quant.Actual365Fixed()
}
frequency={
    "COMPOUNDING":quant.Compounded,
    "CONTINUOUS":quant.Continuous
}
def invoke(input_args=None):    
    main(fx_input_args)
    
def main(input_args):

    # Get dataframes for all parameters required for Heston Calibration
    # The label would be same as seen in the Quant Workbench Data frames defined against the Asset classes FX Menu.See QWB_Demo.ipynb
    spot_prices_df = input_args.get("EXTRACTION_RESULT").get("FX_SPOT_PRICES").get("output")
    surface_prices_df = input_args.get("EXTRACTION_RESULT").get('SF_DOM_FOR_PRICES').get("output")
    cvDom_prices_df = input_args.get("EXTRACTION_RESULT").get('CV_DOM_RATES').get("output")
    cvDom_cal_df=input_args.get("EXTRACTION_RESULT").get('CV_DOM_CALENDAR').get("output")    
    cvDom_convention_df = input_args.get("EXTRACTION_RESULT").get('CV_DOM_CONVENTION').get("output")
    cvFor_prices_df = input_args.get("EXTRACTION_RESULT").get('CV_FOR_RATES').get("output")
    cvFor_cal_df = input_args.get("EXTRACTION_RESULT").get('CV_FOR_CALENDAR').get("output")
    cvFor_convention_df = input_args.get("EXTRACTION_RESULT").get('CV_FOR_CONVENTION').get("output")
    
    #Valuation Date-label should be seen in input parameters section.See QWB_Demo.ipynb
    #TODO: Create a separate utility function for getting the input parameter values
    dtPriceTms = datetime.strptime('23/01/2019', '%d/%m/%Y')
    
    #Convert Pandas date to Quantlib Date format
    ValuationDate = quant.Date(dtPriceTms.day, dtPriceTms.month, dtPriceTms.year)
    #TODO:Revisit next line of code if required
    quant.Settings.instance().evaluationDate = ValuationDate
    
    #Extract spot and daycount conventions from the data frames
    spot=spot_prices_df['Price'][0]
    domdayCountConvention = cvDom_convention_df['DaysPerYearBasis'][0]
    tenor_days=cvDom_prices_df['TenorDays']
    
    #Get Calendar and other convention details for Domestic Curve
    usd_day_count = dayCountsDataset[domdayCountConvention]
    usd_calendar = getCalendar(cvDom_cal_df)#quant.UnitedStates()    
    usd_interpolation = quant.Linear()
    usd_compounding = quant.Compounded
    usd_compoundingFrequency = quant.Continuous

    #Get Calendar and other convention details for Foreign Curve
    jpy_day_count = quant.Actual360()
    jpy_calendar = getCalendar(cvFor_cal_df)#quant.Japan()
        # getCalendar(cvFor_cal_df)
    jpy_interpolation = quant.Linear()
    jpy_compounding = quant.Compounded
    jpy_compoundingFrequency = quant.Continuous

    
    
    # For all tenor days in the surface calculate maturity dates using the valuation date
    datetimedates = [dtPriceTms + timedelta(days=x) for x in tenor_days]
    dates = [(quant.Date(dt.day, dt.month, dt.year)) for dt in datetimedates]

    #Get the rates for the domestic curve for the valuation date
    dom_rates=cvDom_prices_df['Price']
    
    #Get the rates for the Foreign curve for the valuation date
    for_rates=cvFor_prices_df['Price']
    #Bootstrap the zero rate curve for the domestic par rate curve
    for_curve = quant.ZeroCurve(dates, for_rates, usd_day_count, usd_calendar,
                                usd_interpolation, usd_compounding, usd_compoundingFrequency)
    
    #Bootstrap the zero rate curve for the Foreign par rate curve
    dom_curve = quant.ZeroCurve(dates, dom_rates, jpy_day_count, jpy_calendar,
                                jpy_interpolation, jpy_compounding, jpy_compoundingFrequency)
    # term structure definitions
    for_term_structure = quant.YieldTermStructureHandle(for_curve)
    dom_term_structure = quant.YieldTermStructureHandle(dom_curve)

    forward_dates = dates[1:11]

    n = len(forward_dates)
    # Put the rates into a numpy array 
    rates_d=np.array(dom_rates[1:12])
    rates_f=np.array(for_rates[1:12])
    days = np.array(tenor_days[1:12])

    #Difference between domestic and foreign rates.This is required for calculate the 
    #forward FX Rates from spot rates ---incorrect using covered interest rate parity. 
    mu = rates_d - rates_f
    #forward calculation - check the document sent by Michele for fwdrates 
    forwards = spot * np.exp(np.multiply(mu, days / 360))
    
    # Get Deltas and volatilities of the surface.Note:Volatilities are stored in GoldenSource by Delta.
    sf_deltas=surface_prices_df['Delta'].unique()    
    deltas=np.array(sf_deltas)    
    vols = surface_prices_df
    vols = vols.pivot(index='TenorDays', columns='Delta', values='Price')
    vols = np.asarray(vols)
    
    #Calculate the strikes. The volatilities by default stored by delta.Heston calibration requires volatilities to be converted to  strikes
    #TODO: Check formula with Michele- check in the document
    strikes = np.transpose(np.multiply(forwards, np.exp(
        np.multiply(np.transpose(np.multiply((-1) * norm.ppf(np.array(deltas)), (vols / 100))),
                    np.sqrt(days / 360)) + 0.5 * np.multiply(np.transpose(np.power(vols / 100, 2)), days / 360))))

    vols = vols / 100

    listHestonResult = []
    hestonImpliedVols = np.zeros([np.size(vols, 0), np.size(vols, 1)])
    output_list = list()
    for days_idx in range(len(forward_dates)):
        output_dict = dict()
        # initial values for the Heston parameters
        v0 = 0.005
        kappa = 0.5
        theta = 0.01
        rho = 0.1
        sigma = 0.05

        process = quant.HestonProcess(dom_term_structure, for_term_structure,
                                      quant.QuoteHandle(quant.SimpleQuote(spot)),
                                      v0, kappa, theta, sigma, rho)
        model = quant.HestonModel(process)
        # different engines available like Montecarlo engine etc
        engine = quant.AnalyticHestonEngine(model)

        heston_helpers = []

        one_month_idx = days_idx  # 2nd row in vols is for 1 month expiry
        date = forward_dates[one_month_idx]
        
        for j, s in enumerate(strikes[one_month_idx][:]):
            
            t = (date - ValuationDate)
            p = quant.Period(t, quant.Days)
            impliedVol = vols[one_month_idx][j]

            helper = quant.HestonModelHelper(p, usd_calendar, spot, s,
                                             quant.QuoteHandle(quant.SimpleQuote(impliedVol)),
                                             dom_term_structure,
                                             for_term_structure)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            hestonImpliedVols[days_idx][j] = heston_helpers[j].impliedVolatility(heston_helpers[j].modelValue(),
                                                                                 1.e-6, 5000, 0.0001, 1.0)
        
        listHestonResult = []
        #optimization algorithm
        #Check for the Fellar condition here
        lm = quant.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        model.calibrate(heston_helpers, lm, quant.EndCriteria(10000, 100, 1.0e-8, 1.0e-8, 1.0e-8))
        theta, kappa, sigma, rho, v0 = model.params()


        heston_handle = quant.HestonModelHandle(model)
        heston_vol_surface = quant.HestonBlackVolSurface(heston_handle)
        

        graphList=list()
        output,surf_file=plot_vol_surface(heston_vol_surface, dates[0], date, model.params())
        graphList.append(surf_file)
        output, file = fetchResult(dates[0], date, deltas, heston_helpers, model, one_month_idx, strikes)
        output_dict['title'] = str(forward_dates[days_idx])
        output_dict["output"] = output
        
        graphList.append(file)
        output_dict["graph"] = graphList
        output_dict["label"] = "HESTON_CALIBRATION_" + str(forward_dates[days_idx])    
        output_list.append(output_dict)
        

    # Plot the Black volatility surface
    
    implied_vols = quant.Matrix(np.size(vols, 1), np.size(vols, 0))
    for i in range(implied_vols.rows()):
        for j in range(implied_vols.columns()):
            implied_vols[i][j] = vols[j][i]


    return output_list

def plot_vol_surface(vol_surface, valDate, expDate, params, plot_years=np.arange(0.1, 10, 0.1), plot_strikes=np.arange(50, 140, 1)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
#      iplot(data, filename = self._data_filename, asUrl=True, online=False)

    X, Y = np.meshgrid(plot_strikes, plot_years)
    Z = np.array([vol_surface.blackVol(float(y), float(x))
                  for xr, yr in zip(X, Y)
                  for x, y in zip(xr, yr)]
                 ).reshape(len(X), len(X[0]))

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f'Valuation date: {valDate}\nExpiry date: {expDate}\ntheta = {round(params[0], 4)}, '
              f'kappa = {round(params[1], 4)}, sigma = {round(params[2], 4)}, rho = {round(params[3], 4)}, '
              f'v0 = {round(params[4], 4)}',
              fontsize=10)
#     plt.iplot(surf,filename="output" + os.path.sep + str(expDate),asUrl=True,onLine=false)

    output = "plot_vol_surface=" * 80
    output = output + "\n"
    output = output + "Valuation date: " + str(valDate) + "\n"
    output_plot = "output" + os.path.sep + str(expDate) +'_surf.png'
    plt.savefig(output_plot, transparent=True, dpi=500) 
    return  output,output_plot
    
def fetchResult(valDate, expDate, deltas, heston_helpers, model, one_month_idx, strikes):
    theta, kappa, sigma, rho, v0 = model.params()
    output = "=" * 80
    output = output + "\n"
    output = output + "Valuation date: " + str(valDate) + "\n"
    output = output + "Expiration date: " + str(expDate) + "\n"
    output = output + "theta = {theta}, kappa = {kappa}, sigma = {sigma},".format(theta=theta, kappa=kappa,
                                                                                  sigma=sigma) + "\n"
    output = output + "rho = {rho}, v0 = {v0}".format(rho=rho, v0=v0) + "\n"
    avg = 0.0
    output = output + "%5s %15s %15s %15s %20s" % ("Delta",
                                                   "Strikes", "Market Value",
                                                   "Model Value", "Relative Error (%)") + "\n"
    output = output + "=" * 80 + "\n"
    summary = []
    for i, opt in enumerate(heston_helpers):
        err = (opt.modelValue() / opt.marketValue() - 1.0)
        summary.append((
            deltas[i], strikes[one_month_idx][i],
            opt.marketValue(), opt.modelValue(),
            100.0 * (opt.modelValue() / opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg * 100.0 / len(heston_helpers)
    df = pd.DataFrame(
        summary,
        columns=["Deltas", "Strikes", "Market value", "Model value", "Relative error (%)"],
        index=[''] * len(summary))
    output = output + str(df) + "\n"
    output = output + "-" * 80 + "\n"
    output = output + "Average Abs Error (%%) : %5.3f" % (avg) + "\n\n"

    df.set_index('Strikes')[['Market value', 'Model value']].plot(marker='o')
    plt.title(f'Valuation date: {valDate}\nExpiry date: {expDate}\ntheta = {round(theta, 4)}, '
              f'kappa = {round(kappa, 4)}, sigma = {round(sigma, 4)}, rho = {round(rho, 4)}, v0 = {round(v0, 4)}',
              fontsize=10)

    output_plot = "output" + os.path.sep + str(expDate) + '.png'
    plt.savefig(output_plot, transparent=True, dpi=500)

    return output, output_plot


def prepareMatrixPointValueResult(dtPriceTms, listHestonResult, type, mtdiName):
    dictMatrixInstance = {}
    dictMatrixInstance['MATRIX_NME'] = 'Heston FX Vol Smile Parameters'
    dictMatrixInstance['MTDI_NME'] = mtdiName
    listMatrixPointResult = []
    ctr = 0
    for dictResult in listHestonResult:
        dictMatrixPointValue = {}
        dictMatrixPointValue['X_AXIS_TENOR_TYP'] = 'Days'
        dictMatrixPointValue['X_AXIS_VAL_NUM'] = dictResult['days']
        dictMatrixPointValue['X_AXIS_VAL_DTE'] = dtPriceTms
        dictMatrixPointValue['Y_AXIS_VAL_CAMT'] = dictResult[type]
        dictMatrixPointValue['Y_AXIS_VAL_DTE'] = dtPriceTms
        dictMatrixPointValue['POINT_SEQ'] = ctr
        listMatrixPointResult.append(dictMatrixPointValue)
        ctr = ctr + 1
    return dictMatrixInstance, listMatrixPointResult


def printResult(valDate, expDate, deltas, heston_helpers, model, one_month_idx, strikes, vols):
    theta, kappa, sigma, rho, v0 = model.params()
    print("=" * 80)
    print("Valuation date: ", valDate)
    print("Expiration date: ", expDate)
    print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % (theta, kappa, sigma, rho, v0))
    avg = 0.0
    print("%5s %15s %15s %15s %15s %15s %20s" % ("Delta", "Strikes",
                                                 "Market Value", "Model Value",
                                                 "Market Implied Vol", "Model Implied Vol",
                                                 "Relative Error (%)"))
    print("=" * 120)
    summary = []
    for i, opt in enumerate(heston_helpers):
        err = (opt.modelValue() / opt.marketValue() - 1.0)
        summary.append((
            deltas[i], strikes[one_month_idx][i],
            opt.marketValue(), opt.modelValue(),
            vols[i],
            opt.impliedVolatility(opt.modelValue(), 1.e-6, 5000, 0.0001, 1.0),
            100.0 * (opt.modelValue() / opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg * 100.0 / len(heston_helpers)
    df = pd.DataFrame(
        summary,
        columns=["Deltas", "Strikes", "Market value", "Model value", "Market Implied Vol", "Model Implied Vol",
                 "Relative error (%)"],
        index=[''] * len(summary))
    print(df)
    print("-" * 120)
    print("Average Abs Error (%%) : %5.3f" % (avg))

    df.set_index('Strikes')[['Market value', 'Model value']].plot(marker='o')
    plt.title(f'Valuation date: {valDate}\nExpiry date: {expDate}\ntheta = {round(theta, 4)}, '
              f'kappa = {round(kappa, 4)}, sigma = {round(sigma, 4)}, rho = {round(rho, 4)}, v0 = {round(v0, 4)}',
              fontsize=10)
    df.set_index('Strikes')[['Market Implied Vol', 'Model Implied Vol']].plot(marker='o')
    plt.title(f'Valuation date: {valDate}\nExpiry date: {expDate}\ntheta = {round(theta, 4)}, '
              f'kappa = {round(kappa, 4)}, sigma = {round(sigma, 4)}, rho = {round(rho, 4)}, v0 = {round(v0, 4)}',
              fontsize=10)

if __name__ == '__main__':
    main()

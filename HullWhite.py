import QuantLib as ql
import numpy as np
import math
from scipy.stats import norm

# swaption ATM volatilities
def CreateSwaptionVolatilityList():
    vol = [0.1148, 0.1108, 0.1070, 0.1021, 0.1011, 0.1000]
    return vol


class ModelCalibrator:
    def __init__(self, endCriteria):
        self.endCriteria = endCriteria
        self.helpers = []

    def AddCalibrationHelper(self, helper):
        self.helpers.append(helper)

    def Calibrate(self, model, engine, curve, fixedParameters, volType):
        # assign pricing engine to all calibration helpers
        for i in range(len(self.helpers)):
            self.helpers[i].setPricingEngine(engine)
        method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
        if (len(fixedParameters) == 0):
            if (volType == 'Lognormal'):
                model.calibrate(self.helpers, method, self.endCriteria)
            else:
                model.calibrate(self.helpers, method, self.endCriteria, ql.NoConstraint(), [], [True, False])
        else:
            model.calibrate(self.helpers, method, self.endCriteria,
                            ql.NoConstraint(), [], fixedParameters)


# general parameters
tradeDate = ql.Date(15, ql.February, 2002)
ql.Settings.instance().evaluationDate = tradeDate
settlementDate = ql.Date(19, ql.February, 2002)
calendar = ql.TARGET()
fixed_leg_dayCounter = ql.Actual360()
floating_leg_dayCounter = ql.Actual360()

# create market data: term structure and diagonal volatilities
curveHandle = ql.YieldTermStructureHandle(ql.FlatForward(settlementDate, 0.04875825, ql.Actual365Fixed()))
index = ql.USDLibor(ql.Period(3, ql.Months), curveHandle)
fixed_leg_tenor = ql.Period(1, ql.Years)
vol = CreateSwaptionVolatilityList()

# create calibrator object
endCriteria = ql.EndCriteria(1000, 100, 1e-6, 1e-8, 1e-8)
calibrator = ModelCalibrator(endCriteria)
calibrator_normal = ModelCalibrator(endCriteria)

# add swaption helpers to calibrator
for i in range(len(vol)):
    t = i + 1
    tenor = len(vol) - i
    helper = ql.SwaptionHelper(ql.Period(t, ql.Years),
                               ql.Period(tenor, ql.Years),
                               ql.QuoteHandle(ql.SimpleQuote(vol[i])),
                               index,
                               fixed_leg_tenor,
                               fixed_leg_dayCounter,
                               floating_leg_dayCounter,
                               curveHandle)
    calibrator.AddCalibrationHelper(helper)
    helper = ql.SwaptionHelper(ql.Period(t, ql.Years),
                               ql.Period(tenor, ql.Years),
                               ql.QuoteHandle(ql.SimpleQuote(vol[i])),
                               index,
                               fixed_leg_tenor,
                               fixed_leg_dayCounter,
                               floating_leg_dayCounter,
                               curveHandle,
                               ql.BlackCalibrationHelper.RelativePriceError,
                               ql.nullDouble(),
                               1.0,
                               ql.Normal)
    calibrator_normal.AddCalibrationHelper(helper)

# create model and pricing engine, calibrate model and print calibrated parameters
print('case 1: calibrate all involved parameters (HW1F : reversion, sigma)')
model = ql.HullWhite(curveHandle)
engine = ql.JamshidianSwaptionEngine(model)
fixedParameters = []
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print()

print('case 2: calibrate sigma and fix reversion to 0.05')
model = ql.HullWhite(curveHandle, 0.05, 0.0001)
engine = ql.JamshidianSwaptionEngine(model)
fixedParameters = [True, False]
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('fixed reversion: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print()

print('case 3: calibrate reversion and fix sigma to 0.01')
model = ql.HullWhite(curveHandle, 0.05, 0.01)
engine = ql.JamshidianSwaptionEngine(model)
fixedParameters = [False, True]
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion: ' + str(round(model.params()[0], 5)))
print('fixed sigma: ' + str(round(model.params()[1], 5)))
print()

print('case 4: calibrate all involved parameters (HW1F normal vol : reversion, sigma)')
model = ql.HullWhite(curveHandle)
engine = ql.JamshidianSwaptionEngine(model)
fixedParameters = []
calibrator_normal.Calibrate(model, engine, curveHandle, fixedParameters, 'Normal')
print('calibrated reversion: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print()

print('case 5: calibrate all involved parameters (Black-Karasinski with numerical lattice engine for swaptions : '
      'reversion, sigma)')
model = ql.BlackKarasinski(curveHandle)
engine = ql.TreeSwaptionEngine(model, 100)
fixedParameters = []
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print()

print('case 6: calibrate all involved parameters (G2++ with numerical lattice engine for swaptions : reversion, sigma)')
model = ql.G2(curveHandle)
engine = ql.TreeSwaptionEngine(model, 25)
fixedParameters = []
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion 1: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print('calibrated reversion 2: ' + str(round(model.params()[2], 5)))
print('calibrated eta: ' + str(round(model.params()[3], 5)))
print('calibrated correlation: ' + str(round(model.params()[4], 5)))

print('case 7: calibrate all involved parameters (G2++ with Black formula engine : reversion, sigma)')
model = ql.G2(curveHandle)
engine = ql.G2SwaptionEngine(model, 10.0, 400)
fixedParameters = []
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion 1: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print('calibrated reversion 2: ' + str(round(model.params()[2], 5)))
print('calibrated eta: ' + str(round(model.params()[3], 5)))
print('calibrated correlation: ' + str(round(model.params()[4], 5)))

print('case 9: calibrate all involved parameters (G2++ with finite difference engine : reversion, sigma)')
model = ql.G2(curveHandle)
engine = ql.FdG2SwaptionEngine(model)
fixedParameters = []
calibrator.Calibrate(model, engine, curveHandle, fixedParameters, 'Lognormal')
print('calibrated reversion 1: ' + str(round(model.params()[0], 5)))
print('calibrated sigma: ' + str(round(model.params()[1], 5)))
print('calibrated reversion 2: ' + str(round(model.params()[2], 5)))
print('calibrated eta: ' + str(round(model.params()[3], 5)))
print('calibrated correlation: ' + str(round(model.params()[4], 5)))


# Convert lognormal volatilities in normal volatilities
# Exact ATM result
def sigma_normATM(sigma_lnATM, F, T):
    sigma_n = np.sqrt(2 * np.pi / T) * F * (2 * norm.cdf(0.5 * sigma_lnATM * np.sqrt(T) - 1))
    return sigma_n

# General approximation
def sigma_norm(sigma_ln, F, K, T):
    sigma_n = sigma_ln * F * (K / F - 1) / np.log(K / F) * (
                1 - (np.log((K - F) / np.sqrt(K * F * np.log(K / F))) / (np.log(F / K)) ** 2) * sigma_ln * T)
    return sigma_n



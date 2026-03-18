import os
import numpy as np 
import matplotlib.pyplot as plt
import math

import ROOT


#These are data from the CNN model and the SUM (reco) baseline, respectively. They contain the true energies, the predicted energies and the reconstructed energies
#Everything is in GeV
y_true_gev = np.load("y_true_gev.npy")
y_pred_gev = np.load("y_pred_gev.npy")
energies_reco = np.load("energies_reco.npy")


#Plots for visualization of the fits
def plot_all_fits1(y_true, y_pred, energy_bins, name = int):
    if name == 0:
        name = "pred"
    else:
        name = "reco"
    cols = 5
    rows = int(np.ceil(len(energy_bins) / cols))
    c1 = ROOT.TCanvas("c1", "HGCAL Fits", 400 * cols, 400 * rows)
    c1.Divide(cols, rows)

    results = {"E": [], "sig": [], "sig_err": []}

    for i in range(len(energy_bins) - 1):
        e_min, e_max = energy_bins[i], energy_bins[i+1]
        e_center = (e_min + e_max) / 2.0
        mask = (y_true >= e_min) & (y_true < e_max)
        residui_bin = y_pred[mask] - y_true[mask]

        c1.cd(i + 1)

        rms = np.std(residui_bin)
        xmin, xmax = -4*rms, 4*rms
        x = ROOT.RooRealVar("x", "Residuals [GeV]", xmin, xmax)
        hist = ROOT.TH1F(f"h_{i}", f"E = {e_center:.0f} GeV - {name}", 50, xmin, xmax)
        for v in residui_bin: 
            hist.Fill(v)
        datahist = ROOT.RooDataHist(f"dh_{i}", "datahist", ROOT.RooArgList(x), hist)
        
        #I used a DSBC here, residuals at high energies do not have a gaussian shape and the tail effects are more pronounced, so a DSCB is more appropriate to capture the distribution of residuals
        #probably a single tailed CB could be enough, but I wanted to be sure to capture the tails on both sides
        #A better fit must be done to obtain more stable parameters 
        mean = ROOT.RooRealVar("mean", "mean", 0, -5, 5)
        sig = ROOT.RooRealVar("sig", "sigma", rms, 0.1, 15)
        aL = ROOT.RooRealVar("aL", "alphaL", 1.5, 0.1, 5)
        nL = ROOT.RooRealVar("nL", "nL", 5, 1.1, 25)
        aR = ROOT.RooRealVar("aR", "alphaR", 1.5, 0.1, 5)
        nR = ROOT.RooRealVar("nR", "nR", 5, 1.1, 25)

        CB=ROOT.RooCrystalBall("model", "DSCB", x, mean, sig, aL, nL, aR, nR)
        CB.fitTo(datahist, ROOT.RooFit.PrintLevel(-1))
        xframe = x.frame(ROOT.RooFit.Title(f"E = {e_center:.0f} GeV - {name}"))
        datahist.plotOn(xframe, ROOT.RooFit.MarkerSize(0.5))
        CB.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kRed))
        CB.paramOn(xframe, ROOT.RooFit.Parameters(ROOT.RooArgSet(mean, sig, aL, nL, aR, nR)), ROOT.RooFit.Layout(0.55, 0.9, 0.9))
        xframe.Draw()

        results["E"].append(e_center)
        results["sig"].append(sig.getVal())
        results["sig_err"].append(sig.getError())

    c1.Update()
    c1.SaveAs(f"hgcal_grid_fits_{name}.png")
    return results

bins = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310, 340, 370]
res = plot_all_fits1(y_true_gev, y_pred_gev, bins, 0)

res2 = plot_all_fits1(y_true_gev, energies_reco*0.01, bins, 1)

#This is a more correct fit
def plot_all_fits2(y_true, y_pred, energy_bins, name = int):
    if name == 0:
        name = "pred"
    else:
        name = "reco"

    number_of_bins = len(energy_bins) - 1
    results = {"E": [], "sig": [], "sig_err": [], "fit_params": []}

    c = []
    xmin, xmax = -15, 15

    for i in range(number_of_bins):
        c.append(ROOT.TCanvas(f"c_{i}", f"HGCAL bin {i}", 800, 600))
        c[i].cd()
        e_min, e_max = energy_bins[i], energy_bins[i+1]
        e_center = (e_min + e_max) / 2.0
        mask = (y_true >= e_min) & (y_true < e_max)
        residui_bin = y_pred[mask] - y_true[mask]
        rms = np.std(residui_bin)
        x = ROOT.RooRealVar("x", "Residuals [GeV]", xmin, xmax)
        x_arg = ROOT.RooArgSet(x)
        data = ROOT.RooDataSet(f"ds_{i}", "dataset", x_arg)
        bin_width_target = rms / 10.0
        if bin_width_target > 0:
            n_bins_dynamic = int(30.0 / bin_width_target)
            n_bins_dynamic = max(50, min(n_bins_dynamic, 1000)) 
        else:
            n_bins_dynamic = 100
        hist = ROOT.TH1F(f"h_{i}", f"E = {e_center:.0f} GeV - {name}", 50, xmin, xmax)
        for v in residui_bin: 
            hist.Fill(v)
            if xmin <= v <= xmax: # Rimaniamo nel range -15, 15
                x.setVal(v)
                data.add(x_arg)
        datahist = ROOT.RooDataHist(f"dh_{i}", "datahist", ROOT.RooArgList(x), hist)
        
        #I used a DSBC here, residuals at high energies do not have a gaussian shape and the tail effects are more pronounced, so a DSCB is more appropriate to capture the distribution of residuals
        #probably a single tailed CB could be enough, but I wanted to be sure to capture the tails on both sides
        #A better fit must be done to obtain more stable parameters 
        mean = ROOT.RooRealVar("mean", "mean", 0, -5, 5)
        sig = ROOT.RooRealVar("sig", "sigma", rms, 0.1, 15)
        aL = ROOT.RooRealVar("aL", "alphaL", 1.5, 0.1, 5)
        nL = ROOT.RooRealVar("nL", "nL", 17, 17, 20)
        aR = ROOT.RooRealVar("aR", "alphaR", 1.5, 0.1, 5)
        nR = ROOT.RooRealVar("nR", "nR", 17, 17, 20)


        CB=ROOT.RooCrystalBall("model", "DSCB", x, mean, sig, aL, nL, aR, nR)   
        fit_results = CB.fitTo(data, ROOT.RooFit.PrintLevel(1))

        
        xframe = x.frame(ROOT.RooFit.Title(f"E = {e_center:.0f} GeV - {name}"))
        datahist.plotOn(xframe, ROOT.RooFit.MarkerSize(0.5))
        xframe.SetXTitle("Residuals [GeV]")
        xframe.SetYTitle("Events / bin")
        CB.plotOn(xframe, ROOT.RooFit.LineColor(ROOT.kRed))
        CB.paramOn(xframe, ROOT.RooFit.Parameters(ROOT.RooArgSet(mean, sig)), ROOT.RooFit.Layout(0.55, 0.9, 0.9))
        xframe.Draw()

        results["fit_params"].append({
            "e_center": e_center,
            "mean":     mean.getVal(),
            "mean_err": mean.getError(),
            "sig":      sig.getVal(),
            "sig_err":  sig.getError(),
            "aL":       aL.getVal(),
            "nL":       nL.getVal(),
            "aR":       aR.getVal(),
            "nR":       nR.getVal(),
            "n_events": len(residui_bin),
        })

        results["E"].append(e_center)
        results["sig"].append(sig.getVal())
        results["sig_err"].append(sig.getError())
        c[i].Update()
        c[i].SaveAs(f"hgcal_grid_fits_{name}_{i}.png")
        c[i].Close()
    return results

#These are the results array that are used for the final plots.
res3 = plot_all_fits2(y_true_gev, y_pred_gev, bins, 0)

res4 = plot_all_fits2(y_true_gev, energies_reco*0.01, bins, 1)


#Visualization of the fits for all energy bins in a single plot, to better compare the shapes of the residuals distributions across 
#different energy ranges and between the CNN model and the SUM baseline. This is not a fit, but just a visualization of the fits obtained with the previous function.
def plot_stacked_fits(results_pred, xmin=-15, xmax=15):
    _root_objects = []  

    canvas = ROOT.TCanvas("c_stacked", "Stacked DSCB fits", 1200, 700)
    canvas.cd()
    _root_objects.append(canvas)

    palette = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange+1,
               ROOT.kMagenta, ROOT.kCyan+1, ROOT.kViolet, ROOT.kTeal,
               ROOT.kPink+1, ROOT.kAzure+2]

    legend = ROOT.TLegend(0.79, 0.12, 0.99, 0.92)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.018)
    legend.SetFillStyle(0)
    _root_objects.append(legend)

    first = True
    for idx, params in enumerate(results_pred["fit_params"]):
        x    = ROOT.RooRealVar(f"x_{idx}",    "Residuals [GeV]", xmin, xmax)
        mean = ROOT.RooRealVar(f"mean_{idx}", "mean",  params["mean"], -5,  5)
        sig  = ROOT.RooRealVar(f"sig_{idx}",  "sigma", params["sig"],  0.1, 15)
        aL   = ROOT.RooRealVar(f"aL_{idx}",   "aL",    params["aL"],   0.1, 10)
        nL   = ROOT.RooRealVar(f"nL_{idx}",   "nL",    params["nL"],   1,   50)
        aR   = ROOT.RooRealVar(f"aR_{idx}",   "aR",    params["aR"],   0.1, 10)
        nR   = ROOT.RooRealVar(f"nR_{idx}",   "nR",    params["nR"],   1,   50)

        _root_objects.extend([x, mean, sig, aL, nL, aR, nR])

        for var in [mean, sig, aL, nL, aR, nR]:
            var.setConstant(True)

        CB = ROOT.RooCrystalBall(f"cb_{idx}", "DSCB", x, mean, sig,
                                 aL, nL, aR, nR)
        _root_objects.append(CB)


        xframe = x.frame(ROOT.RooFit.Range(xmin, xmax))
        _root_objects.append(xframe)

        norm_curve = CB.createHistogram(f"Distributions for all Energy ranges", x,
                                        ROOT.RooFit.Binning(500))
        _root_objects.append(norm_curve)
        norm_curve.SetTitle("Distribution of residuals for all energy ranges - CNN Model")
        norm_curve.Scale(1.0 / norm_curve.GetMaximum())
        norm_curve.SetLineColor(palette[idx % len(palette)])
        norm_curve.SetLineWidth(2)
        norm_curve.SetStats(False)
        norm_curve.GetXaxis().SetTitle("Residuals [GeV]")
        norm_curve.GetYaxis().SetTitle("a.u.")
        norm_curve.GetYaxis().SetRangeUser(0, 1.3)

        opt = "HIST L SAME" if not first else "HIST L"
        norm_curve.Draw(opt)
        first = False

        legend.AddEntry(norm_curve, f"E = {params['e_center']:.0f} GeV", "l")

    legend.Draw()

    line = ROOT.TLine(0, 0, 0, 1.3)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kGray+2)
    line.Draw()
    _root_objects.append(line)

    canvas.SaveAs("stacked_fits.png")
    canvas.SaveAs("stacked_fits.pdf")
    return canvas, _root_objects

stacked_canvas, stacked_curves = plot_stacked_fits(res3)


#Single plot for resolution, first version
def fit_resolution_linear(results, name = int):
    if name == 0:
        name = "pred"
    else:
        name = "reco"
    

    n_points = len(results["E"])
    x = np.array(results["E"], dtype='float64')
    x_val = 1.0 / np.sqrt(x) 
    y_val = np.array(results["sig"], dtype='float64') / x
    y_err = np.array(results["sig_err"], dtype='float64') / x

    graph = ROOT.TGraphErrors(n_points, x_val, y_val, np.zeros(n_points), y_err)
    graph.SetTitle("HGCAL Energy Resolution; 1/#sqrt{E} [GeV^{-1/2}];#sigma(E)/E")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.2)

    func = ROOT.TF1("res_func", "sqrt(([0]*x)^2 + [1]^2)", 0, 0.3)
    func.SetParameters(0.20, 0.01)
    func.SetParNames("S", "C")
    func.SetLineColor(ROOT.kRed)
    func.SetLineStyle(2)
    fit_res = graph.Fit(func, "S")
    c_final = ROOT.TCanvas("c_final", "Final Resolution Fit", 800, 800)
    graph.Draw("AP")
    
    # Aggiunta legenda e statistiche
    legend = ROOT.TLegend(0.5, 0.7, 0.88, 0.88)
    legend.AddEntry(graph, "Data", "pe")
    legend.AddEntry(func, f"Fit: S={func.GetParameter(0)*100:.2f}%, C={func.GetParameter(1)*100:.2f}%", "l")
    legend.Draw()
    
    c_final.Update()
    c_final.Draw()
    c_final.SaveAs(f"final_resolution_fit_{name}.png")

    return func.GetParameter(0), func.GetParameter(1)

fit_resolution_linear(res, 0)
fit_resolution_linear(res2, 1)



#Single plot for resolution, second version
def fit_resolution_curve(results, name = int):
    if name == 0:
        name = "pred"
    else:
        name = "reco"

    n_points = len(results["E"])
    x = np.array(results["E"], dtype='float64')
    x_val = x 
    y_val = np.array(results["sig"], dtype='float64') / x
    y_err = np.array(results["sig_err"], dtype='float64') / x

    graph = ROOT.TGraphErrors(n_points, x_val, y_val, np.zeros(n_points), y_err)
    graph.SetTitle("HGCAL Energy Resolution; 1/#sqrt{E} [GeV^{-1/2}];#sigma(E)/E")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.2)

    func = ROOT.TF1("res_func", "sqrt([0]*[0] / x + [1]*[1])", 5, 400)
    func.SetParameters(0.20, 0.01)
    func.SetParNames("S", "C")
    func.SetLineColor(ROOT.kRed)
    func.SetLineStyle(2)
    fit_res = graph.Fit(func, "S")
    c_final_res = ROOT.TCanvas("c_final", "Final Resolution Fit", 800, 800)
    graph.Draw("AP")
    
    legend = ROOT.TLegend(0.5, 0.7, 0.88, 0.88)
    legend.AddEntry(graph, "Data", "pe")
    legend.AddEntry(func, f"Fit: S={func.GetParameter(0)*100:.2f}%, C={func.GetParameter(1)*100:.2f}%", "l")
    legend.Draw()
    
    c_final_res.Update()
    c_final_res.Draw()
    c_final_res.SaveAs(f"final_resolution_fit_res_{name}.png")

    return func.GetParameter(0), func.GetParameter(1)

fit_resolution_curve(res, 0)
fit_resolution_curve(res2, 1)


#Final comparison plot, first version
def fit_resolution_linear_final(results1, results2):
    


    x1 = np.array(results1["E"], dtype='float64')
    x_val1 = 1.0 / np.sqrt(x1) 
    y_val1 = np.array(results1["sig"], dtype='float64') / x1
    y_err1 = np.array(results1["sig_err"], dtype='float64') / x1
    
    graph1 = ROOT.TGraphErrors(len(x1), x_val1, y_val1, np.zeros(len(x1)), y_err1)
    graph1.SetMarkerStyle(20)
    graph1.SetMarkerSize(1.2)
    graph1.SetMarkerColor(ROOT.kBlack)
    graph1.SetLineColor(ROOT.kBlack)


    x2 = np.array(results2["E"], dtype='float64')
    x_val2 = 1.0 / np.sqrt(x2)
    y_val2 = np.array(results2["sig"], dtype='float64') / x2
    y_err2 = np.array(results2["sig_err"], dtype='float64') / x2
    
    graph2 = ROOT.TGraphErrors(len(x2), x_val2, y_val2, np.zeros(len(x2)), y_err2)
    graph2.SetMarkerStyle(21)
    graph2.SetMarkerSize(1.2)
    graph2.SetMarkerColor(ROOT.kRed)
    graph2.SetLineColor(ROOT.kRed)

    mg = ROOT.TMultiGraph()
    mg.Add(graph1, "P")
    mg.Add(graph2, "P")
    mg.SetTitle("HGCAL Energy Resolution;1/#sqrt{E} [GeV^{-1/2}];#sigma(E)/E")

    func1 = ROOT.TF1("func1", "sqrt(([0]*x)^2 + [1]^2)", 0, 0.3)
    func1.SetParameters(0.20, 0.01)
    func1.SetLineColor(ROOT.kBlack)
    func1.SetLineStyle(2)
    graph1.Fit(func1, "S")

    func2 = ROOT.TF1("func2", "sqrt(([0]*x)^2 + [1]^2)", 0, 0.3)
    func2.SetParameters(0.20, 0.01)
    func2.SetLineColor(ROOT.kRed)
    func2.SetLineStyle(2)
    graph2.Fit(func2, "S")

    c1 = ROOT.TCanvas("c_final", "Final Resolution Fit", 1280, 1000)
    c1.SetGrid()
    mg.Draw("AP")


    legend = ROOT.TLegend(0.4, 0.7, 0.88, 0.88)
    legend.SetBorderSize(1)
    legend.AddEntry(graph1, "CNN Model", "pe")
    legend.AddEntry(func1, f"Fit CNN: S={func1.GetParameter(0)*100:.2f}%, C={func1.GetParameter(1)*100:.2f}%", "l")
    legend.AddEntry(graph2, "SUM Baseline", "pe")
    legend.AddEntry(func2, f"Fit SUM: S={func2.GetParameter(0)*100:.2f}%, C={func2.GetParameter(1)*100:.2f}%", "l")
    legend.Draw()

    c1.Update()
    c1.SaveAs("resolution_comparison.png")

    return func1.GetParameter(0), func1.GetParameter(1), func2.GetParameter(0), func2.GetParameter(1)

fit_resolution_linear_final(res3, res4)

#Final comparison plot, second version
def fit_resolution_comparison_curve(results1, results2):


    x1 = np.array(results1["E"], dtype='float64')
    y_val1 = np.array(results1["sig"], dtype='float64') / x1
    y_err1 = np.array(results1["sig_err"], dtype='float64') / x1
    
    graph1 = ROOT.TGraphErrors(len(x1), x1, y_val1, np.zeros(len(x1)), y_err1)
    graph1.SetMarkerStyle(20)
    graph1.SetMarkerSize(1.2)
    graph1.SetMarkerColor(ROOT.kBlack)
    graph1.SetLineColor(ROOT.kBlack)


    x2 = np.array(results2["E"], dtype='float64')
    y_val2 = np.array(results2["sig"], dtype='float64') / x2
    y_err2 = np.array(results2["sig_err"], dtype='float64') / x2
    
    graph2 = ROOT.TGraphErrors(len(x2), x2, y_val2, np.zeros(len(x2)), y_err2)
    graph2.SetMarkerStyle(21)
    graph2.SetMarkerSize(1.2)
    graph2.SetMarkerColor(ROOT.kRed)
    graph2.SetLineColor(ROOT.kRed)

    mg = ROOT.TMultiGraph()
    mg.Add(graph1, "P")
    mg.Add(graph2, "P")
    mg.SetTitle("Comparison HGCAL Energy Resolution;E [GeV];#sigma(E)/E")

    func1 = ROOT.TF1("func1", "sqrt(([0]*[0])/x + [1]*[1])", 5, 400)
    func1.SetParameters(0.20, 0.01)
    func1.SetLineColor(ROOT.kBlack)
    func1.SetLineStyle(2)
    graph1.Fit(func1, "SR")

    func2 = ROOT.TF1("func2", "sqrt(([0]*[0])/x + [1]*[1])", 5, 400)
    func2.SetParameters(0.20, 0.01)
    func2.SetLineColor(ROOT.kRed)
    func2.SetLineStyle(2)
    graph2.Fit(func2, "SR")

    c1 = ROOT.TCanvas("c_comp_curva", "Resolution Comparison Curve", 1280, 1000)
    c1.SetGrid()
    mg.Draw("AP")
    
    legend = ROOT.TLegend(0.45, 0.65, 0.88, 0.88)
    legend.SetBorderSize(1)
    legend.AddEntry(graph1, "CNN Model", "pe")
    legend.AddEntry(func1, f"Fit CNN: S={func1.GetParameter(0)*100:.2f}%, C={func1.GetParameter(1)*100:.2f}%", "l")
    legend.AddEntry(graph2, "SUM Baseline", "pe")
    legend.AddEntry(func2, f"Fit SUM: S={func2.GetParameter(0)*100:.2f}%, C={func2.GetParameter(1)*100:.2f}%", "l")
    legend.Draw()

    c1.Update()
    c1.SaveAs("resolution_comparison_curva.png")

    return func1.GetParameters(), func2.GetParameters()

fit_resolution_comparison_curve(res3, res4)


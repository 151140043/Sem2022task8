import pandas as pd
from scipy.stats import pearsonr

true_data = pd.read_csv("mid_data/mid_trail_data.csv", header=0)
predict_data = pd.read_csv("result/1.csv", header=0)
pearsonO = pearsonr(true_data["Overall"], predict_data["Overall"])
pearsonG = pearsonr(true_data['Geography'], predict_data['Geography'])
pearsonT = pearsonr(true_data['Time'], predict_data['Time'])
pearsonTo = pearsonr(true_data['Tone'], predict_data['Tone'])
pearsonE = pearsonr(true_data['Entities'], predict_data['Entities'])
pearsonN = pearsonr(true_data['Narrative'], predict_data['Narrative'])
pearsonS = pearsonr(true_data['Style'], predict_data['Style'])
print("Overall:", pearsonO)
print("Time:", pearsonT)
print("Geography:", pearsonG)
print("Tone:", pearsonTo)
print("Entities:", pearsonE)
print("Narrative:", pearsonN)
print("Style:", pearsonS)




import pandas as pd
import pyreadstat

# Load the SAS dataset
df, meta = pyreadstat.read_sas7bdat("./NHATS_R13_Final_Release_SAS/NHATS_Round_13_SP_File.sas7bdat")

# Save as CSV
df.to_csv("NHATS_Round_13_SP_File.csv", index=False)

# Display first few rows to check
print(df.head())

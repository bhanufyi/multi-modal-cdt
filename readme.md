### Multi Modal LLM's Test for CDT Score

I have downloaded dataset from [NHATS](https://nhats.org/researcher/data-access)

Used Round 13 for the current test. I have used 50 samples at present. 

Demographic and Health data are present in SAS format. So I had to convert them to XLSX format, I have selected only records where CDT score is present.

I have used the following columns for Demographics 
`R13 WB3 AGE YOU FEEL MOST OF TIME`
`R13 D RACE AND HISPANIC ETHNICITY WHEN ADDED`
`R13 D LONGEST OCCUPATION CATEGORY`

and clock drawing image. 

Tried with both with demographic context and only image. 

Accuracy:
With Patient Info Accuracy: 0.74 
Without Patient Info Accuracy: 0.732

Most of the weightage is given to the image. Demographic helping a bit, but not significant. Again the test size is small, we might have to select more fields for context and more samples to generalize the results.

Next step would be to try out with other Multi Modal LLM's.
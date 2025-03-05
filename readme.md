### Multi-Modal LLM Test for CDT Scoring

### **Dataset & Setup**

- Downloaded data from [NHATS](https://nhats.org/researcher/data-access)
- Used **Round 13** dataset for the current test
- Selected **50 samples** for evaluation

### **Data Processing**

- **Demographic & Health Data:** Originally in SAS format, converted to XLSX
- **Filtering Criteria:** Selected only records where **CDT score** is available
- **Demographic Features Used:**
    - `R13 WB3 AGE YOU FEEL MOST OF TIME`
    - `R13 D RACE AND HISPANIC ETHNICITY WHEN ADDED`
    - `R13 D LONGEST OCCUPATION CATEGORY`
- **Testing Approaches:**
    - **With demographic context + clock drawing image**
    - **Using only the clock drawing image**

### **Results**

| Model Configuration | Accuracy |
| --- | --- |
| With Patient Info | **0.74** |
| Without Patient Info | **0.732** |


With even distribution.
Version	RMSE	Regression Accuracy	
With Patient Info	0.966091783079296	0.693333333333333	
No Patient Info	1.01653004546513	0.706666666666667	
- **Key Findings:**
    - **Clock drawing image** contributes the most to scoring accuracy
    - **Demographic information* is not highly significant
    - **Test size is small**, further validation required with a larger dataset
    - Additional demographic fields might improve results

### **Next Steps**

- **Expand dataset** to include more samples for better generalization
- **Experiment with additional demographic fields** for improved context
- **Test with other multi-modal LLMs** to compare performance
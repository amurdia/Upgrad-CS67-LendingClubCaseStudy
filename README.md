# Upgrad-CS67-LendingClubCaseStudy

```python
# Importing the expected libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Loading dataset to pandas dataframe
file_path = r"./loan.csv"
df = pd.read_csv(file_path)
# Checking few values for the given dataset
df.head()
```

    /var/folders/m6/80bkjjx93lxgkzttt0bnl9740000gq/T/ipykernel_46461/1838048065.py:3: DtypeWarning: Columns (47) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv(file_path)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 111 columns</p>
</div>




```python
# Checking the detailed info of the given dataset
df.info(verbose=1)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39717 entries, 0 to 39716
    Data columns (total 111 columns):
     #    Column                          Dtype  
    ---   ------                          -----  
     0    id                              int64  
     1    member_id                       int64  
     2    loan_amnt                       int64  
     3    funded_amnt                     int64  
     4    funded_amnt_inv                 float64
     5    term                            object 
     6    int_rate                        object 
     7    installment                     float64
     8    grade                           object 
     9    sub_grade                       object 
     10   emp_title                       object 
     11   emp_length                      object 
     12   home_ownership                  object 
     13   annual_inc                      float64
     14   verification_status             object 
     15   issue_d                         object 
     16   loan_status                     object 
     17   pymnt_plan                      object 
     18   url                             object 
     19   desc                            object 
     20   purpose                         object 
     21   title                           object 
     22   zip_code                        object 
     23   addr_state                      object 
     24   dti                             float64
     25   delinq_2yrs                     int64  
     26   earliest_cr_line                object 
     27   inq_last_6mths                  int64  
     28   mths_since_last_delinq          float64
     29   mths_since_last_record          float64
     30   open_acc                        int64  
     31   pub_rec                         int64  
     32   revol_bal                       int64  
     33   revol_util                      object 
     34   total_acc                       int64  
     35   initial_list_status             object 
     36   out_prncp                       float64
     37   out_prncp_inv                   float64
     38   total_pymnt                     float64
     39   total_pymnt_inv                 float64
     40   total_rec_prncp                 float64
     41   total_rec_int                   float64
     42   total_rec_late_fee              float64
     43   recoveries                      float64
     44   collection_recovery_fee         float64
     45   last_pymnt_d                    object 
     46   last_pymnt_amnt                 float64
     47   next_pymnt_d                    object 
     48   last_credit_pull_d              object 
     49   collections_12_mths_ex_med      float64
     50   mths_since_last_major_derog     float64
     51   policy_code                     int64  
     52   application_type                object 
     53   annual_inc_joint                float64
     54   dti_joint                       float64
     55   verification_status_joint       float64
     56   acc_now_delinq                  int64  
     57   tot_coll_amt                    float64
     58   tot_cur_bal                     float64
     59   open_acc_6m                     float64
     60   open_il_6m                      float64
     61   open_il_12m                     float64
     62   open_il_24m                     float64
     63   mths_since_rcnt_il              float64
     64   total_bal_il                    float64
     65   il_util                         float64
     66   open_rv_12m                     float64
     67   open_rv_24m                     float64
     68   max_bal_bc                      float64
     69   all_util                        float64
     70   total_rev_hi_lim                float64
     71   inq_fi                          float64
     72   total_cu_tl                     float64
     73   inq_last_12m                    float64
     74   acc_open_past_24mths            float64
     75   avg_cur_bal                     float64
     76   bc_open_to_buy                  float64
     77   bc_util                         float64
     78   chargeoff_within_12_mths        float64
     79   delinq_amnt                     int64  
     80   mo_sin_old_il_acct              float64
     81   mo_sin_old_rev_tl_op            float64
     82   mo_sin_rcnt_rev_tl_op           float64
     83   mo_sin_rcnt_tl                  float64
     84   mort_acc                        float64
     85   mths_since_recent_bc            float64
     86   mths_since_recent_bc_dlq        float64
     87   mths_since_recent_inq           float64
     88   mths_since_recent_revol_delinq  float64
     89   num_accts_ever_120_pd           float64
     90   num_actv_bc_tl                  float64
     91   num_actv_rev_tl                 float64
     92   num_bc_sats                     float64
     93   num_bc_tl                       float64
     94   num_il_tl                       float64
     95   num_op_rev_tl                   float64
     96   num_rev_accts                   float64
     97   num_rev_tl_bal_gt_0             float64
     98   num_sats                        float64
     99   num_tl_120dpd_2m                float64
     100  num_tl_30dpd                    float64
     101  num_tl_90g_dpd_24m              float64
     102  num_tl_op_past_12m              float64
     103  pct_tl_nvr_dlq                  float64
     104  percent_bc_gt_75                float64
     105  pub_rec_bankruptcies            float64
     106  tax_liens                       float64
     107  tot_hi_cred_lim                 float64
     108  total_bal_ex_mort               float64
     109  total_bc_limit                  float64
     110  total_il_high_credit_limit      float64
    dtypes: float64(74), int64(13), object(24)
    memory usage: 33.6+ MB


# Data Cleanup


```python
# Evaluating the fields with nan values
missing_value_percentage=df.isna().sum()*100/df.shape[0]
missing_value_percentage.sort_values(ascending=False)
```




    verification_status_joint    100.0
    annual_inc_joint             100.0
    mo_sin_old_rev_tl_op         100.0
    mo_sin_old_il_acct           100.0
    bc_util                      100.0
                                 ...  
    delinq_amnt                    0.0
    policy_code                    0.0
    earliest_cr_line               0.0
    delinq_2yrs                    0.0
    id                             0.0
    Length: 111, dtype: float64




```python
# Filtering the fields with all nan records
missing_value_percentage[missing_value_percentage == 100]
```




    mths_since_last_major_derog       100.0
    annual_inc_joint                  100.0
    dti_joint                         100.0
    verification_status_joint         100.0
    tot_coll_amt                      100.0
    tot_cur_bal                       100.0
    open_acc_6m                       100.0
    open_il_6m                        100.0
    open_il_12m                       100.0
    open_il_24m                       100.0
    mths_since_rcnt_il                100.0
    total_bal_il                      100.0
    il_util                           100.0
    open_rv_12m                       100.0
    open_rv_24m                       100.0
    max_bal_bc                        100.0
    all_util                          100.0
    total_rev_hi_lim                  100.0
    inq_fi                            100.0
    total_cu_tl                       100.0
    inq_last_12m                      100.0
    acc_open_past_24mths              100.0
    avg_cur_bal                       100.0
    bc_open_to_buy                    100.0
    bc_util                           100.0
    mo_sin_old_il_acct                100.0
    mo_sin_old_rev_tl_op              100.0
    mo_sin_rcnt_rev_tl_op             100.0
    mo_sin_rcnt_tl                    100.0
    mort_acc                          100.0
    mths_since_recent_bc              100.0
    mths_since_recent_bc_dlq          100.0
    mths_since_recent_inq             100.0
    mths_since_recent_revol_delinq    100.0
    num_accts_ever_120_pd             100.0
    num_actv_bc_tl                    100.0
    num_actv_rev_tl                   100.0
    num_bc_sats                       100.0
    num_bc_tl                         100.0
    num_il_tl                         100.0
    num_op_rev_tl                     100.0
    num_rev_accts                     100.0
    num_rev_tl_bal_gt_0               100.0
    num_sats                          100.0
    num_tl_120dpd_2m                  100.0
    num_tl_30dpd                      100.0
    num_tl_90g_dpd_24m                100.0
    num_tl_op_past_12m                100.0
    pct_tl_nvr_dlq                    100.0
    percent_bc_gt_75                  100.0
    tot_hi_cred_lim                   100.0
    total_bal_ex_mort                 100.0
    total_bc_limit                    100.0
    total_il_high_credit_limit        100.0
    dtype: float64




```python
# Dropping all fields with no data
df.drop(df.columns[missing_value_percentage == 100].tolist(), axis='columns', inplace=True)
```


```python
# Evaluating the fields with nan values in the updated dataframe
missing_value_percentage=df.isna().sum()*100/df.shape[0]
missing_value_percentage.sort_values(ascending=False)
```




    next_pymnt_d                  97.129693
    mths_since_last_record        92.985372
    mths_since_last_delinq        64.662487
    desc                          32.585543
    emp_title                      6.191303
    emp_length                     2.706650
    pub_rec_bankruptcies           1.754916
    last_pymnt_d                   0.178765
    chargeoff_within_12_mths       0.140998
    collections_12_mths_ex_med     0.140998
    revol_util                     0.125891
    tax_liens                      0.098195
    title                          0.027696
    last_credit_pull_d             0.005036
    home_ownership                 0.000000
    int_rate                       0.000000
    out_prncp_inv                  0.000000
    total_pymnt                    0.000000
    total_pymnt_inv                0.000000
    total_rec_prncp                0.000000
    total_rec_int                  0.000000
    total_rec_late_fee             0.000000
    recoveries                     0.000000
    collection_recovery_fee        0.000000
    term                           0.000000
    last_pymnt_amnt                0.000000
    initial_list_status            0.000000
    funded_amnt_inv                0.000000
    policy_code                    0.000000
    application_type               0.000000
    acc_now_delinq                 0.000000
    funded_amnt                    0.000000
    delinq_amnt                    0.000000
    loan_amnt                      0.000000
    out_prncp                      0.000000
    total_acc                      0.000000
    annual_inc                     0.000000
    addr_state                     0.000000
    verification_status            0.000000
    issue_d                        0.000000
    loan_status                    0.000000
    pymnt_plan                     0.000000
    url                            0.000000
    sub_grade                      0.000000
    purpose                        0.000000
    zip_code                       0.000000
    dti                            0.000000
    installment                    0.000000
    delinq_2yrs                    0.000000
    earliest_cr_line               0.000000
    inq_last_6mths                 0.000000
    member_id                      0.000000
    grade                          0.000000
    open_acc                       0.000000
    pub_rec                        0.000000
    revol_bal                      0.000000
    id                             0.000000
    dtype: float64




```python
# Based on the observation of above data, dropping all fields with more than 90% nan data
df.drop(df.columns[missing_value_percentage > 90].tolist(), axis='columns', inplace=True)
df.info(verbose=1)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39717 entries, 0 to 39716
    Data columns (total 55 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   id                          39717 non-null  int64  
     1   member_id                   39717 non-null  int64  
     2   loan_amnt                   39717 non-null  int64  
     3   funded_amnt                 39717 non-null  int64  
     4   funded_amnt_inv             39717 non-null  float64
     5   term                        39717 non-null  object 
     6   int_rate                    39717 non-null  object 
     7   installment                 39717 non-null  float64
     8   grade                       39717 non-null  object 
     9   sub_grade                   39717 non-null  object 
     10  emp_title                   37258 non-null  object 
     11  emp_length                  38642 non-null  object 
     12  home_ownership              39717 non-null  object 
     13  annual_inc                  39717 non-null  float64
     14  verification_status         39717 non-null  object 
     15  issue_d                     39717 non-null  object 
     16  loan_status                 39717 non-null  object 
     17  pymnt_plan                  39717 non-null  object 
     18  url                         39717 non-null  object 
     19  desc                        26775 non-null  object 
     20  purpose                     39717 non-null  object 
     21  title                       39706 non-null  object 
     22  zip_code                    39717 non-null  object 
     23  addr_state                  39717 non-null  object 
     24  dti                         39717 non-null  float64
     25  delinq_2yrs                 39717 non-null  int64  
     26  earliest_cr_line            39717 non-null  object 
     27  inq_last_6mths              39717 non-null  int64  
     28  mths_since_last_delinq      14035 non-null  float64
     29  open_acc                    39717 non-null  int64  
     30  pub_rec                     39717 non-null  int64  
     31  revol_bal                   39717 non-null  int64  
     32  revol_util                  39667 non-null  object 
     33  total_acc                   39717 non-null  int64  
     34  initial_list_status         39717 non-null  object 
     35  out_prncp                   39717 non-null  float64
     36  out_prncp_inv               39717 non-null  float64
     37  total_pymnt                 39717 non-null  float64
     38  total_pymnt_inv             39717 non-null  float64
     39  total_rec_prncp             39717 non-null  float64
     40  total_rec_int               39717 non-null  float64
     41  total_rec_late_fee          39717 non-null  float64
     42  recoveries                  39717 non-null  float64
     43  collection_recovery_fee     39717 non-null  float64
     44  last_pymnt_d                39646 non-null  object 
     45  last_pymnt_amnt             39717 non-null  float64
     46  last_credit_pull_d          39715 non-null  object 
     47  collections_12_mths_ex_med  39661 non-null  float64
     48  policy_code                 39717 non-null  int64  
     49  application_type            39717 non-null  object 
     50  acc_now_delinq              39717 non-null  int64  
     51  chargeoff_within_12_mths    39661 non-null  float64
     52  delinq_amnt                 39717 non-null  int64  
     53  pub_rec_bankruptcies        39020 non-null  float64
     54  tax_liens                   39678 non-null  float64
    dtypes: float64(19), int64(13), object(23)
    memory usage: 16.7+ MB



```python
# Analysing remaining fields in the dataset
for col in df.columns:
    print(df[col].describe())
    print("==================================\n\n")
```

    count    3.971700e+04
    mean     6.831319e+05
    std      2.106941e+05
    min      5.473400e+04
    25%      5.162210e+05
    50%      6.656650e+05
    75%      8.377550e+05
    max      1.077501e+06
    Name: id, dtype: float64
    ==================================
    
    
    count    3.971700e+04
    mean     8.504636e+05
    std      2.656783e+05
    min      7.069900e+04
    25%      6.667800e+05
    50%      8.508120e+05
    75%      1.047339e+06
    max      1.314167e+06
    Name: member_id, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean     11219.443815
    std       7456.670694
    min        500.000000
    25%       5500.000000
    50%      10000.000000
    75%      15000.000000
    max      35000.000000
    Name: loan_amnt, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean     10947.713196
    std       7187.238670
    min        500.000000
    25%       5400.000000
    50%       9600.000000
    75%      15000.000000
    max      35000.000000
    Name: funded_amnt, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean     10397.448868
    std       7128.450439
    min          0.000000
    25%       5000.000000
    50%       8975.000000
    75%      14400.000000
    max      35000.000000
    Name: funded_amnt_inv, dtype: float64
    ==================================
    
    
    count          39717
    unique             2
    top        36 months
    freq           29096
    Name: term, dtype: object
    ==================================
    
    
    count      39717
    unique       371
    top       10.99%
    freq         956
    Name: int_rate, dtype: object
    ==================================
    
    
    count    39717.000000
    mean       324.561922
    std        208.874874
    min         15.690000
    25%        167.020000
    50%        280.220000
    75%        430.780000
    max       1305.190000
    Name: installment, dtype: float64
    ==================================
    
    
    count     39717
    unique        7
    top           B
    freq      12020
    Name: grade, dtype: object
    ==================================
    
    
    count     39717
    unique       35
    top          B3
    freq       2917
    Name: sub_grade, dtype: object
    ==================================
    
    
    count       37258
    unique      28820
    top       US Army
    freq          134
    Name: emp_title, dtype: object
    ==================================
    
    
    count         38642
    unique           11
    top       10+ years
    freq           8879
    Name: emp_length, dtype: object
    ==================================
    
    
    count     39717
    unique        5
    top        RENT
    freq      18899
    Name: home_ownership, dtype: object
    ==================================
    
    
    count    3.971700e+04
    mean     6.896893e+04
    std      6.379377e+04
    min      4.000000e+03
    25%      4.040400e+04
    50%      5.900000e+04
    75%      8.230000e+04
    max      6.000000e+06
    Name: annual_inc, dtype: float64
    ==================================
    
    
    count            39717
    unique               3
    top       Not Verified
    freq             16921
    Name: verification_status, dtype: object
    ==================================
    
    
    count      39717
    unique        55
    top       Dec-11
    freq        2260
    Name: issue_d, dtype: object
    ==================================
    
    
    count          39717
    unique             3
    top       Fully Paid
    freq           32950
    Name: loan_status, dtype: object
    ==================================
    
    
    count     39717
    unique        1
    top           n
    freq      39717
    Name: pymnt_plan, dtype: object
    ==================================
    
    
    count                                                 39717
    unique                                                39717
    top       https://lendingclub.com/browse/loanDetail.acti...
    freq                                                      1
    Name: url, dtype: object
    ==================================
    
    
    count     26775
    unique    26526
    top            
    freq        210
    Name: desc, dtype: object
    ==================================
    
    
    count                  39717
    unique                    14
    top       debt_consolidation
    freq                   18641
    Name: purpose, dtype: object
    ==================================
    
    
    count                  39706
    unique                 19615
    top       Debt Consolidation
    freq                    2184
    Name: title, dtype: object
    ==================================
    
    
    count     39717
    unique      823
    top       100xx
    freq        597
    Name: zip_code, dtype: object
    ==================================
    
    
    count     39717
    unique       50
    top          CA
    freq       7099
    Name: addr_state, dtype: object
    ==================================
    
    
    count    39717.000000
    mean        13.315130
    std          6.678594
    min          0.000000
    25%          8.170000
    50%         13.400000
    75%         18.600000
    max         29.990000
    Name: dti, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean         0.146512
    std          0.491812
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max         11.000000
    Name: delinq_2yrs, dtype: float64
    ==================================
    
    
    count      39717
    unique       526
    top       Nov-98
    freq         370
    Name: earliest_cr_line, dtype: object
    ==================================
    
    
    count    39717.000000
    mean         0.869200
    std          1.070219
    min          0.000000
    25%          0.000000
    50%          1.000000
    75%          1.000000
    max          8.000000
    Name: inq_last_6mths, dtype: float64
    ==================================
    
    
    count    14035.000000
    mean        35.900962
    std         22.020060
    min          0.000000
    25%         18.000000
    50%         34.000000
    75%         52.000000
    max        120.000000
    Name: mths_since_last_delinq, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean         9.294408
    std          4.400282
    min          2.000000
    25%          6.000000
    50%          9.000000
    75%         12.000000
    max         44.000000
    Name: open_acc, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean         0.055065
    std          0.237200
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          4.000000
    Name: pub_rec, dtype: float64
    ==================================
    
    
    count     39717.000000
    mean      13382.528086
    std       15885.016641
    min           0.000000
    25%        3703.000000
    50%        8850.000000
    75%       17058.000000
    max      149588.000000
    Name: revol_bal, dtype: float64
    ==================================
    
    
    count     39667
    unique     1089
    top          0%
    freq        977
    Name: revol_util, dtype: object
    ==================================
    
    
    count    39717.000000
    mean        22.088828
    std         11.401709
    min          2.000000
    25%         13.000000
    50%         20.000000
    75%         29.000000
    max         90.000000
    Name: total_acc, dtype: float64
    ==================================
    
    
    count     39717
    unique        1
    top           f
    freq      39717
    Name: initial_list_status, dtype: object
    ==================================
    
    
    count    39717.000000
    mean        51.227887
    std        375.172839
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       6311.470000
    Name: out_prncp, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean        50.989768
    std        373.824457
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       6307.370000
    Name: out_prncp_inv, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean     12153.596544
    std       9042.040766
    min          0.000000
    25%       5576.930000
    50%       9899.640319
    75%      16534.433040
    max      58563.679930
    Name: total_pymnt, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean     11567.149118
    std       8942.672613
    min          0.000000
    25%       5112.310000
    50%       9287.150000
    75%      15798.810000
    max      58563.680000
    Name: total_pymnt_inv, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean      9793.348813
    std       7065.522127
    min          0.000000
    25%       4600.000000
    50%       8000.000000
    75%      13653.260000
    max      35000.020000
    Name: total_rec_prncp, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean      2263.663172
    std       2608.111964
    min          0.000000
    25%        662.180000
    50%       1348.910000
    75%       2833.400000
    max      23563.680000
    Name: total_rec_int, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean         1.363015
    std          7.289979
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max        180.200000
    Name: total_rec_late_fee, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean        95.221624
    std        688.744771
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max      29623.350000
    Name: recoveries, dtype: float64
    ==================================
    
    
    count    39717.000000
    mean        12.406112
    std        148.671593
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       7002.190000
    Name: collection_recovery_fee, dtype: float64
    ==================================
    
    
    count      39646
    unique       101
    top       May-16
    freq        1256
    Name: last_pymnt_d, dtype: object
    ==================================
    
    
    count    39717.000000
    mean      2678.826162
    std       4447.136012
    min          0.000000
    25%        218.680000
    50%        546.140000
    75%       3293.160000
    max      36115.200000
    Name: last_pymnt_amnt, dtype: float64
    ==================================
    
    
    count      39715
    unique       106
    top       May-16
    freq       10308
    Name: last_credit_pull_d, dtype: object
    ==================================
    
    
    count    39661.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: collections_12_mths_ex_med, dtype: float64
    ==================================
    
    
    count    39717.0
    mean         1.0
    std          0.0
    min          1.0
    25%          1.0
    50%          1.0
    75%          1.0
    max          1.0
    Name: policy_code, dtype: float64
    ==================================
    
    
    count          39717
    unique             1
    top       INDIVIDUAL
    freq           39717
    Name: application_type, dtype: object
    ==================================
    
    
    count    39717.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: acc_now_delinq, dtype: float64
    ==================================
    
    
    count    39661.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: chargeoff_within_12_mths, dtype: float64
    ==================================
    
    
    count    39717.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: delinq_amnt, dtype: float64
    ==================================
    
    
    count    39020.000000
    mean         0.043260
    std          0.204324
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max          2.000000
    Name: pub_rec_bankruptcies, dtype: float64
    ==================================
    
    
    count    39678.0
    mean         0.0
    std          0.0
    min          0.0
    25%          0.0
    50%          0.0
    75%          0.0
    max          0.0
    Name: tax_liens, dtype: float64
    ==================================
    
    



```python
# Based on the analysis of fields above, `term` field needs to be convert to integer.
# Checking the variation in values
df.term.value_counts(ascending=False)
```




    term
    36 months    29096
    60 months    10621
    Name: count, dtype: int64




```python
# Converting `term` to integer
df['term'] = df.term.apply(lambda x: int(x.replace(' months', '')))
df.term.value_counts(ascending=False)
```




    term
    36    29096
    60    10621
    Name: count, dtype: int64




```python
# Based on the analysis of fields above, `int_rate` field needs to be convert to float.
# Checking the variation in values
df.int_rate.value_counts(ascending=False)
```




    int_rate
    10.99%    956
    13.49%    826
    11.49%    825
    7.51%     787
    7.88%     725
             ... 
    18.36%      1
    16.96%      1
    16.15%      1
    16.01%      1
    17.44%      1
    Name: count, Length: 371, dtype: int64




```python
# Converting `int_rate` to float
df['int_rate'] = df.int_rate.apply(lambda x: float(str(x).replace('%', '')))
df['int_rate'] = df.int_rate.astype(float)
df.int_rate.value_counts(ascending=False)
```




    int_rate
    10.99    956
    13.49    826
    11.49    825
    7.51     787
    7.88     725
            ... 
    18.36      1
    16.96      1
    16.15      1
    16.01      1
    17.44      1
    Name: count, Length: 371, dtype: int64




```python
df.int_rate.describe()
```




    count    39717.000000
    mean        12.021177
    std          3.724825
    min          5.420000
    25%          9.250000
    50%         11.860000
    75%         14.590000
    max         24.590000
    Name: int_rate, dtype: float64




```python
# Based on the analysis of fields above, `emp_length` field needs to be convert to integer.
# Checking the variation in values
df.emp_length.value_counts(ascending=False)
```




    emp_length
    10+ years    8879
    < 1 year     4583
    2 years      4388
    3 years      4095
    4 years      3436
    5 years      3282
    1 year       3240
    6 years      2229
    7 years      1773
    8 years      1479
    9 years      1258
    Name: count, dtype: int64




```python
# Extracting integer values for `emp_length`
def extract_num(val):
  val = str(val).strip()

  if val.startswith("10+"):
    return 10
  elif val.startswith("< 1") or val == "nan":
    return 0
  else:
    return int(val.split(' ')[0])

df['emp_length'] = df.emp_length.apply(extract_num)
```


```python
df.emp_length.value_counts()
```




    emp_length
    10    8879
    0     5658
    2     4388
    3     4095
    4     3436
    5     3282
    1     3240
    6     2229
    7     1773
    8     1479
    9     1258
    Name: count, dtype: int64




```python
# Importing datetime library
from datetime import datetime

# Custoom method for converting string/null to datetime object
def convert_to_datetime_obj(x):
    if x == "epoch":
        return datetime(1970, 1, 1, 0, 0, 0)

    return datetime.strptime(x, '%b-%y')

# Filling null values in datetime attributes to the string 'epoch'. And then converting date fields from string to datetime object.
df['issue_d'] = df.issue_d.fillna("epoch")
df['issue_d'] = df.issue_d.apply(convert_to_datetime_obj)

df['earliest_cr_line'] = df.earliest_cr_line.fillna("epoch")
df['earliest_cr_line'] = df.earliest_cr_line.apply(convert_to_datetime_obj)

df['last_credit_pull_d'] = df.last_credit_pull_d.fillna("epoch")
df['last_credit_pull_d'] = df.last_credit_pull_d.apply(convert_to_datetime_obj)

df['last_pymnt_d'] = df.last_pymnt_d.fillna("epoch")
df['last_pymnt_d'] = df.last_pymnt_d.apply(convert_to_datetime_obj)
```


```python
# Extracting loan issue year and month from `issue_d` field
from datetime import datetime
df["loan_issue_year"] = pd.DatetimeIndex(df.issue_d).year
df["loan_issue_month"] = pd.DatetimeIndex(df.issue_d).month
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>loan_issue_year</th>
      <th>loan_issue_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>INDIVIDUAL</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>




```python
# Based on analysis of fields above, there are a bunch of fields with exactly the same data for all records
# collections_12_mths_ex_med: 0
# policy_code: 1
# acc_now_delinq: 0
# chargeoff_within_12_mths: 0
# delinq_amnt: 0
# tax_liens: 0
# application_type: INDIVIDUAL
# pymnt_plan: n

# Dropping these columns
df.drop(['collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens', 'application_type', 'pymnt_plan'], axis='columns', inplace=True)
df.info(verbose=1)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39717 entries, 0 to 39716
    Data columns (total 49 columns):
     #   Column                   Non-Null Count  Dtype         
    ---  ------                   --------------  -----         
     0   id                       39717 non-null  int64         
     1   member_id                39717 non-null  int64         
     2   loan_amnt                39717 non-null  int64         
     3   funded_amnt              39717 non-null  int64         
     4   funded_amnt_inv          39717 non-null  float64       
     5   term                     39717 non-null  int64         
     6   int_rate                 39717 non-null  float64       
     7   installment              39717 non-null  float64       
     8   grade                    39717 non-null  object        
     9   sub_grade                39717 non-null  object        
     10  emp_title                37258 non-null  object        
     11  emp_length               39717 non-null  int64         
     12  home_ownership           39717 non-null  object        
     13  annual_inc               39717 non-null  float64       
     14  verification_status      39717 non-null  object        
     15  issue_d                  39717 non-null  datetime64[ns]
     16  loan_status              39717 non-null  object        
     17  url                      39717 non-null  object        
     18  desc                     26775 non-null  object        
     19  purpose                  39717 non-null  object        
     20  title                    39706 non-null  object        
     21  zip_code                 39717 non-null  object        
     22  addr_state               39717 non-null  object        
     23  dti                      39717 non-null  float64       
     24  delinq_2yrs              39717 non-null  int64         
     25  earliest_cr_line         39717 non-null  datetime64[ns]
     26  inq_last_6mths           39717 non-null  int64         
     27  mths_since_last_delinq   14035 non-null  float64       
     28  open_acc                 39717 non-null  int64         
     29  pub_rec                  39717 non-null  int64         
     30  revol_bal                39717 non-null  int64         
     31  revol_util               39667 non-null  object        
     32  total_acc                39717 non-null  int64         
     33  initial_list_status      39717 non-null  object        
     34  out_prncp                39717 non-null  float64       
     35  out_prncp_inv            39717 non-null  float64       
     36  total_pymnt              39717 non-null  float64       
     37  total_pymnt_inv          39717 non-null  float64       
     38  total_rec_prncp          39717 non-null  float64       
     39  total_rec_int            39717 non-null  float64       
     40  total_rec_late_fee       39717 non-null  float64       
     41  recoveries               39717 non-null  float64       
     42  collection_recovery_fee  39717 non-null  float64       
     43  last_pymnt_d             39717 non-null  datetime64[ns]
     44  last_pymnt_amnt          39717 non-null  float64       
     45  last_credit_pull_d       39717 non-null  datetime64[ns]
     46  pub_rec_bankruptcies     39020 non-null  float64       
     47  loan_issue_year          39717 non-null  int32         
     48  loan_issue_month         39717 non-null  int32         
    dtypes: datetime64[ns](4), float64(17), int32(2), int64(12), object(14)
    memory usage: 14.5+ MB



```python
# Dropping unnecessary or unstructured text attributes
df.drop(columns=['desc', 'title', 'id', 'member_id', 'url', 'zip_code', 'sub_grade', 'emp_title'], axis='columns', inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>last_credit_pull_d</th>
      <th>pub_rec_bankruptcies</th>
      <th>loan_issue_year</th>
      <th>loan_issue_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>10</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>...</td>
      <td>863.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2015-01-01</td>
      <td>171.62</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>0</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>...</td>
      <td>435.17</td>
      <td>0.00</td>
      <td>117.08</td>
      <td>1.11</td>
      <td>2013-04-01</td>
      <td>119.66</td>
      <td>2013-09-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>...</td>
      <td>605.67</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014-06-01</td>
      <td>649.91</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>...</td>
      <td>2214.92</td>
      <td>16.97</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2015-01-01</td>
      <td>357.48</td>
      <td>2016-04-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>1</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>...</td>
      <td>1037.39</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2016-05-01</td>
      <td>67.79</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>




```python
# Imputing null values in `pub_rec_bankruptcies` to 0
df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)  # Assume no bankruptcy if missing
```


```python
# Checking for remaining missing values
df.isna().sum().sort_values(ascending=False)
```




    mths_since_last_delinq     25682
    revol_util                    50
    loan_amnt                      0
    total_rec_prncp                0
    total_acc                      0
    initial_list_status            0
    out_prncp                      0
    out_prncp_inv                  0
    total_pymnt                    0
    total_pymnt_inv                0
    total_rec_int                  0
    pub_rec                        0
    total_rec_late_fee             0
    recoveries                     0
    collection_recovery_fee        0
    last_pymnt_d                   0
    last_pymnt_amnt                0
    last_credit_pull_d             0
    pub_rec_bankruptcies           0
    loan_issue_year                0
    revol_bal                      0
    open_acc                       0
    funded_amnt                    0
    annual_inc                     0
    funded_amnt_inv                0
    term                           0
    int_rate                       0
    installment                    0
    grade                          0
    emp_length                     0
    home_ownership                 0
    verification_status            0
    inq_last_6mths                 0
    issue_d                        0
    loan_status                    0
    purpose                        0
    addr_state                     0
    dti                            0
    delinq_2yrs                    0
    earliest_cr_line               0
    loan_issue_month               0
    dtype: int64




```python
# Final shape of the dataset
df.shape
```




    (39717, 41)




```python
# Check all available columns
print(df.columns)

# Check first few rows of the dataframe
df.head()
```

    Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc',
           'verification_status', 'issue_d', 'loan_status', 'purpose',
           'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
           'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
           'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
           'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
           'last_credit_pull_d', 'pub_rec_bankruptcies', 'loan_issue_year',
           'loan_issue_month'],
          dtype='object')





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>last_credit_pull_d</th>
      <th>pub_rec_bankruptcies</th>
      <th>loan_issue_year</th>
      <th>loan_issue_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>10</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>...</td>
      <td>863.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2015-01-01</td>
      <td>171.62</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>0</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>...</td>
      <td>435.17</td>
      <td>0.00</td>
      <td>117.08</td>
      <td>1.11</td>
      <td>2013-04-01</td>
      <td>119.66</td>
      <td>2013-09-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>...</td>
      <td>605.67</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2014-06-01</td>
      <td>649.91</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>...</td>
      <td>2214.92</td>
      <td>16.97</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2015-01-01</td>
      <td>357.48</td>
      <td>2016-04-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>1</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>...</td>
      <td>1037.39</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2016-05-01</td>
      <td>67.79</td>
      <td>2016-05-01</td>
      <td>0.0</td>
      <td>2011</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



# UNIVARIATE ANALYSIS

## 1. Loan Amount
The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.

Distribution of Loan amount can provide insights into the market demand


```python
df.loan_amnt.value_counts(ascending=False)
```




    loan_amnt
    10000    2833
    12000    2334
    5000     2051
    6000     1908
    15000    1895
             ... 
    22875       1
    8175        1
    19475       1
    21225       1
    22550       1
    Name: count, Length: 885, dtype: int64




```python
sns.histplot(df.loan_amnt)
plt.show()
```


    
![png](output_30_0.png)
    


## 2. Interest Rate
Interest Rate on the loan

Distribution of interest rate showcases the following:
- Highest profitability for the bank.
- Acceptable range of interest rates.
- High risk loans (when compared with other variables)
- Low risk loans (when compared with other variables)


```python
df.int_rate.value_counts(ascending=False)
```




    int_rate
    10.99    956
    13.49    826
    11.49    825
    7.51     787
    7.88     725
            ... 
    18.36      1
    16.96      1
    16.15      1
    16.01      1
    17.44      1
    Name: count, Length: 371, dtype: int64




```python
sns.histplot(df.int_rate)
plt.show()
```


    
![png](output_33_0.png)
    


## 3. Annual Income
The self-reported annual income provided by the borrower during registration.

Distribution of annual income indicates the existing customer segment and can point to potential customers to target.


```python
df.annual_inc.describe()
```




    count    3.971700e+04
    mean     6.896893e+04
    std      6.379377e+04
    min      4.000000e+03
    25%      4.040400e+04
    50%      5.900000e+04
    75%      8.230000e+04
    max      6.000000e+06
    Name: annual_inc, dtype: float64




```python
sns.histplot(df.annual_inc, bins=50)
plt.show()
```


    
![png](output_36_0.png)
    



```python
sns.boxplot(df.annual_inc)
plt.show()
```


    
![png](output_37_0.png)
    



```python
# Since the customers with extremely high income seem to be outliers, we can possibly exclude them for our analysis.
df_excluding_hii = df[df.annual_inc < 1000000]
sns.boxplot(df_excluding_hii.annual_inc)
plt.show()
```


    
![png](output_38_0.png)
    


## 3. Annual Income
The self-reported annual income provided by the borrower during registration.

This is an important metric for evaluating the borrower's ability to repay. Distribution of this metric points to the risk factor in repayment of existing loans.


```python
sns.histplot(df.dti)
plt.show()
```


    
![png](output_40_0.png)
    



```python
sns.boxplot(df.dti)
plt.show()
```


    
![png](output_41_0.png)
    


## 4. Address State
The state provided by the borrower in the loan application.

This metric provides following insights:
- State-wise demand for loans.
- Highest current cutomer base.
- Lowest current customer base.


```python
plt.figure(figsize=(15,5))
sns.histplot(df.addr_state)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_43_0.png)
    


## 5. Home Ownership
The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.

This metric helps in idenitfying customers who do not have any other long tenure loan or liability. Also, it points to the customers with higher financial stability.


```python
sns.histplot(df.home_ownership)
plt.show()
```


    
![png](output_45_0.png)
    


## 6. Loan Status
Current status of the loan.

This showcases the existing customer with unpaid loans. This indirectly also points to the total funds still getting rotated in market versus liquidity with the bank for offering loans to potential customers.


```python
sns.histplot(df.loan_status)
plt.show()
```


    
![png](output_47_0.png)
    


## 7. Grade
LC assigned loan grade.

This metric defines range of high risk loans.


```python
# Bar plot for 'grade'
plt.figure(figsize=(10,6))
sns.countplot(x='grade', data=df, order=df['grade'].value_counts().index)
plt.title('Distribution of Loan Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()
```


    
![png](output_49_0.png)
    


## 8. Employment Length
Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.

Distribution of this metric gives insight into customers with varied employment lengths and possibly the ability to repay the loan.


```python
# Assuming 'emp_length' signifies employment status
plt.figure(figsize=(10,6))
sns.countplot(x='emp_length', data=df, order=df['emp_length'].value_counts().index)
plt.title('Employment Length Distribution')
plt.xlabel('Employment Length')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_51_0.png)
    


## 9. Purpose
A category provided by the borrower for the loan request.

Shows distribution of market demand for different types of loans.


```python
# Bar plot for 'purpose'
plt.figure(figsize=(12,6))
sns.countplot(x='purpose', data=df, order=df['purpose'].value_counts().index)
plt.title('Distribution of Loan Purpose')
plt.xlabel('Purpose')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_53_0.png)
    


## 10. Loan Term
The number of payments on the loan. Values are in months and can be either 36 or 60.

Shows distribution of loan term.


```python
# Assuming 'term' signifies loan term
plt.figure(figsize=(10,6))
sns.countplot(x='term', data=df, order=df['term'].value_counts().index)
plt.title('Loan Term Distribution')
plt.xlabel('Loan Term (months)')
plt.ylabel('Count')
plt.show()
```


    
![png](output_55_0.png)
    


## 11. Loan to Income Ratio
This is a derived metric obtained as a ratio of Loan Amount and Income. A higher ratio may signal that a borrower is over-leveraged, making it a strong predictor of default risk.


```python
# Creating new metrices
# 1. Create Loan to Income Ratio
df['loan_to_income_ratio'] = df['loan_amnt'] / df['annual_inc']

# Visualize Loan to Income Ratio
plt.figure(figsize=(10,6))
sns.histplot(df['loan_to_income_ratio'], bins=30, kde=True)
plt.title('Loan to Income Ratio Distribution')
plt.xlabel('Loan to Income Ratio')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_57_0.png)
    


## 12. Credit Score Proxy
Calculating a FICO proxy based on existing credit variables. Credit scores often correlate with risk levels, and this binning will help identify risk patterns more clearly.


```python
# 2. Calculating a FICO proxy based on existing credit variables
# Using 'delinq_2yrs', 'inq_last_6mths', 'revol_util', 'total_acc' to create an aggregated credit score proxy


# Converting 'revol_util' from string to numeric, and filling missing values
df['revol_util'] = pd.to_numeric(df['revol_util'].str.replace('%', ''), errors='coerce')
df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

# Create an aggregated credit score proxy based on available variables (this can be adjusted)
df['credit_score_proxy'] = (100 - df['delinq_2yrs']*20 - df['inq_last_6mths']*10 + df['total_acc']*2) * (100 - df['revol_util'])

# Credit Score Proxy Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['credit_score_proxy'], bins=30, kde=True)
plt.title('Credit Score Proxy Distribution')
plt.xlabel('Credit Score Proxy')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_59_0.png)
    


## 13. Missed Payments
If the data includes missed payments, create a metric that counts them. More missed payments would strongly indicate the risk of future default.


```python
# 3. Calculating missed payments based on 'loan_status' and 'total_rec_late_fee'
# If 'loan_status' is 'Charged Off' or 'Default', mark it as missed payment.
df['missed_payments'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

# If total late fee > 0, mark as missed payments
df['missed_payments'] = df.apply(lambda x: 1 if x['total_rec_late_fee'] > 0 else x['missed_payments'], axis=1)

# Missed Payments Distribution
plt.figure(figsize=(10,6))
sns.countplot(x='missed_payments', data=df)
plt.title('Missed Payments Distribution')
plt.xlabel('Missed Payments (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
```


    
![png](output_61_0.png)
    



```python
# Credit Utilization Rate
df['credit_utilization'] = df['revol_bal'] / (df['open_acc'] + 1)

# Installment-to-Income Ratio
df['installment_to_income_ratio'] = df['installment'] / (df['annual_inc'] / 12)

# Delinquency Risk Index
df['delinquency_risk_index'] = df['delinq_2yrs'] + df['pub_rec']

# Loan Payment to Principal Ratio
df['loan_payment_to_principal_ratio'] = df['total_pymnt'] / df['loan_amnt']

# Loan Principal Recovery Rate
df['loan_principal_recovery_rate'] = df['recoveries'] / df['loan_amnt']

# Loan Charge-Off Rate
df['loan_charge_off_rate'] = df['collection_recovery_fee'] / df['loan_amnt']

# Employment Length as Categorical
df['emp_length_cat'] = pd.cut(df['emp_length'], bins=[-1, 1, 5, 10, 20], labels=["<1yr", "1-5yr", "5-10yr", "10+yr"])

# Convert loan_status to binary where 'Charged Off' is 1 and others are 0
df['loan_status_binary'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>credit_score_proxy</th>
      <th>missed_payments</th>
      <th>credit_utilization</th>
      <th>installment_to_income_ratio</th>
      <th>delinquency_risk_index</th>
      <th>loan_payment_to_principal_ratio</th>
      <th>loan_principal_recovery_rate</th>
      <th>loan_charge_off_rate</th>
      <th>emp_length_cat</th>
      <th>loan_status_binary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000</td>
      <td>5000</td>
      <td>4975.0</td>
      <td>36</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>10</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>...</td>
      <td>1760.4</td>
      <td>0</td>
      <td>3412.000000</td>
      <td>0.081435</td>
      <td>0</td>
      <td>1.172631</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5-10yr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>0</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>...</td>
      <td>5254.8</td>
      <td>1</td>
      <td>421.750000</td>
      <td>0.023932</td>
      <td>0</td>
      <td>0.403484</td>
      <td>0.046832</td>
      <td>0.000444</td>
      <td>&lt;1yr</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>...</td>
      <td>150.0</td>
      <td>0</td>
      <td>985.333333</td>
      <td>0.082595</td>
      <td>0</td>
      <td>1.252361</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5-10yr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>36</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>10</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>...</td>
      <td>12956.0</td>
      <td>1</td>
      <td>508.909091</td>
      <td>0.082759</td>
      <td>0</td>
      <td>1.223189</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5-10yr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000</td>
      <td>3000</td>
      <td>3000.0</td>
      <td>60</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>1</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>...</td>
      <td>8113.6</td>
      <td>0</td>
      <td>1736.437500</td>
      <td>0.010169</td>
      <td>0</td>
      <td>1.171110</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>&lt;1yr</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>




```python
# Set up the visual style
sns.set(style="whitegrid")
```

# Segmented Analysis


```python
# Check all available columns
print(df.columns)
```

    Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc',
           'verification_status', 'issue_d', 'loan_status', 'purpose',
           'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
           'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
           'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
           'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
           'last_credit_pull_d', 'pub_rec_bankruptcies', 'loan_issue_year',
           'loan_issue_month', 'loan_to_income_ratio', 'credit_score_proxy',
           'missed_payments', 'credit_utilization', 'installment_to_income_ratio',
           'delinquency_risk_index', 'loan_payment_to_principal_ratio',
           'loan_principal_recovery_rate', 'loan_charge_off_rate',
           'emp_length_cat', 'loan_status_binary'],
          dtype='object')


## 1. By Categorised Employment Length (emp_length_cat):
Analyze key variables like loan_amnt, int_rate, and term to see how loan characteristics differ by employment length category and whether certain employment lengths are more prone to default.


```python
# 1. Segment by Employment Length (emp_length_cat)
# Analyze loan amount, interest rate, and term for each employment length category

plt.figure(figsize=(10, 6))
sns.boxplot(x='emp_length_cat', y='loan_amnt', hue='loan_status', data=df)
plt.title('Loan Amount by Employment Length Category and Loan Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='emp_length_cat', y='int_rate', hue='loan_status', data=df)
plt.title('Interest Rate by Employment Length Category and Loan Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='term', hue='emp_length_cat', data=df)
plt.title('Loan Term Distribution by Employment Length Category')
plt.show()
```


    
![png](output_67_0.png)
    



    
![png](output_67_1.png)
    



    
![png](output_67_2.png)
    



## 2. By Income Level (annual_inc):
Segment the annual_inc column into bins (e.g., low, medium, high income) and examine variables like loan_amnt, loan_term, and int_rate to check for any trends in defaults based on income levels.


```python
# 2. Segment by Income Level (annual_inc)
# Create income bins and analyze loan characteristics
df['income_level'] = pd.cut(df['annual_inc'], 
                                    bins=[0, 50000, 100000, 150000, 200000, df['annual_inc'].max()], 
                                    labels=['Low', 'Medium', 'High', 'Very High', 'Top 1%'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='income_level', y='loan_amnt', hue='loan_status', data=df)
plt.title('Loan Amount by Income Level and Loan Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='income_level', y='int_rate', hue='loan_status', data=df)
plt.title('Interest Rate by Income Level and Loan Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='term', hue='income_level', data=df)
plt.title('Loan Term Distribution by Income Level')
plt.show()
```


    
![png](output_69_0.png)
    



    
![png](output_69_1.png)
    



    
![png](output_69_2.png)
    


## 3. By Region (addr_state):
Break down by region (US states) to identify if certain regions have a higher proportion of risky loans. Analyze loan_amnt and int_rate for regional patterns.


```python
# 3. Segment by Region (addr_state)
# Analyze loan amount and interest rate by region

plt.figure(figsize=(12, 8))
sns.boxplot(x='addr_state', y='loan_amnt', hue='loan_status', data=df)
plt.title('Loan Amount by Region and Loan Status')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='addr_state', y='int_rate', hue='loan_status', data=df)
plt.title('Interest Rate by Region and Loan Status')
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_71_0.png)
    



    
![png](output_71_1.png)
    


## 4. By Loan Purpose (purpose):
Examine how loan characteristics vary based on the purpose of the loan and whether specific purposes are linked to higher default rates.


```python
# 4. Segment by Loan Purpose (purpose)
# Analyze loan amount, interest rate, and term by loan purpose

plt.figure(figsize=(12, 8))
sns.boxplot(x='purpose', y='loan_amnt', hue='loan_status', data=df)
plt.title('Loan Amount by Loan Purpose and Loan Status')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='purpose', y='int_rate', hue='loan_status', data=df)
plt.title('Interest Rate by Loan Purpose and Loan Status')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='term', hue='purpose', data=df)
plt.title('Loan Term Distribution by Loan Purpose')
plt.xticks(rotation=90)
plt.show()

```


    
![png](output_73_0.png)
    



    
![png](output_73_1.png)
    



    
![png](output_73_2.png)
    


# Bivariate Analysis

## 1. Credit Utilization vs. Loan Status


```python
# . Credit Utilization vs. Loan Status
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='credit_utilization', hue='loan_status', kde=True, bins=50)
plt.title('Credit Utilization Distribution for Loan Status')
plt.show()
```


    
![png](output_76_0.png)
    


## 2. Installment-to-Income Ratio vs. Loan Status


```python
# . Installment-to-Income Ratio vs. Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='installment_to_income_ratio', data=df)
plt.title('Installment-to-Income Ratio vs Loan Status')
plt.show()
```


    
![png](output_78_0.png)
    


## 3. Delinquency Risk Index vs. Loan Status


```python
# . Delinquency Risk Index vs. Loan Status
plt.figure(figsize=(10, 6))
sns.countplot(x='delinquency_risk_index', hue='loan_status', data=df)
plt.title('Delinquency Risk Index by Loan Status')
plt.show()
```


    
![png](output_80_0.png)
    


## 4. Loan Payment to Principal Ratio vs. Loan Status


```python
# . Loan Payment to Principal Ratio vs. Loan Status
# Loan Payment to Principal Ratio by Loan Status - Boxplot (Better Visualization)
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='loan_payment_to_principal_ratio', data=df)
plt.title('Loan Payment to Principal Ratio by Loan Status')
plt.show()
```


    
![png](output_82_0.png)
    


## 5. Loan Principal Recovery Rate vs. Loan Status


```python
# . Loan Principal Recovery Rate vs. Loan Status
plt.figure(figsize=(10, 6))
sns.violinplot(x='loan_status', y='loan_principal_recovery_rate', data=df)
plt.title('Loan Principal Recovery Rate by Loan Status')
plt.show()
```


    
![png](output_84_0.png)
    


## 6. Loan Charge-Off Rate by Grade


```python
# . Loan Charge-Off Rate by Grade
plt.figure(figsize=(10, 6))
sns.barplot(x='grade', y='loan_charge_off_rate', data=df)
plt.title('Loan Charge-Off Rate by Loan Grade')
plt.show()
```


    
![png](output_86_0.png)
    


## 7. Loan Amount vs Loan Status


```python
# . Loan Amount vs Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title('Loan Amount vs Loan Status')
plt.ylabel('Loan Amount')
plt.xlabel('Loan Status')
plt.show()
```


    
![png](output_88_0.png)
    


## 8. Debt-to-Income (DTI) Ratio vs Loan Status


```python
# . Debt-to-Income (DTI) Ratio vs Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='dti', data=df)
plt.title('DTI vs Loan Status')
plt.ylabel('Debt-to-Income Ratio')
plt.xlabel('Loan Status')
plt.show()
```


    
![png](output_90_0.png)
    



```python
## 9. Interest Rate vs Loan Status
```


```python
# . Interest Rate vs Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='int_rate', data=df)
plt.title('Interest Rate vs Loan Status')
plt.ylabel('Interest Rate (%)')
plt.xlabel('Loan Status')
plt.show()
```


    
![png](output_92_0.png)
    


## 9. Employment Length vs Loan Status


```python
# . Employment Length vs Loan Status
plt.figure(figsize=(10, 6))
sns.countplot(x='emp_length_cat', hue='loan_status', data=df)
plt.title('Employment Length vs Loan Status')
plt.ylabel('Count')
plt.xlabel('Employment Length Category')
plt.show()
```


    
![png](output_94_0.png)
    


## 10. Annual Income vs Loan Status


```python
# . Annual Income vs Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='annual_inc', data=df)
plt.title('Annual Income vs Loan Status')
plt.ylabel('Annual Income')
plt.yscale('log')  # Log scale to better display high-income values
plt.xlabel('Loan Status')
plt.show()
```


    
![png](output_96_0.png)
    


## 11. Home Ownership vs Loan Status


```python
# . Home Ownership vs Loan Status
plt.figure(figsize=(10, 6))
sns.countplot(x='home_ownership', hue='loan_status', data=df)
plt.title('Home Ownership vs Loan Status')
plt.ylabel('Count')
plt.xlabel('Home Ownership')
plt.show()
```


    
![png](output_98_0.png)
    


## 12. Loan Amount vs Annual Income (colored by Loan Status)


```python
# . Loan Amount vs Annual Income (colored by Loan Status)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='annual_inc', y='loan_amnt', hue='loan_status', data=df, alpha=0.7)
plt.title('Loan Amount vs Annual Income (colored by Loan Status)')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.yscale('log')
plt.xscale('log')
plt.show()
```


    
![png](output_100_0.png)
    


## 13. Interest Rate vs Credit Utilization (colored by Loan Status)


```python
# . Interest Rate vs Credit Utilization (colored by Loan Status)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='credit_utilization', y='int_rate', hue='loan_status', data=df, alpha=0.7)
plt.title('Interest Rate vs Credit Utilization (colored by Loan Status)')
plt.xlabel('Credit Utilization (%)')
plt.ylabel('Interest Rate (%)')
plt.show()
```


    
![png](output_102_0.png)
    


## 14. Loan Term vs Loan Amount (colored by Loan Status)


```python
# . Loan Term vs Loan Amount (colored by Loan Status)
plt.figure(figsize=(10, 6))
sns.boxplot(x='term', y='loan_amnt', hue='loan_status', data=df)
plt.title('Loan Term vs Loan Amount (colored by Loan Status)')
plt.ylabel('Loan Amount')
plt.xlabel('Loan Term')
plt.show()
```


    
![png](output_104_0.png)
    


## 15. Employment Status vs Loan Purpose (colored by Loan Status)


```python
# . Employment Status vs Loan Purpose (colored by Loan Status)
plt.figure(figsize=(10, 6))
sns.countplot(x='purpose', hue='loan_status', data=df)
plt.title('Loan Purpose vs Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_106_0.png)
    


## 16. Delinquency Risk Index vs Loan Status


```python
# Plot delinquency risk index and loan status
plt.figure(figsize=(10, 6))
sns.violinplot(x='loan_status_binary', y='delinquency_risk_index', data=df, inner=None)
plt.yscale('log')
plt.title('Delinquency Risk Index vs Loan Status (Log Scale)')
plt.xlabel('Loan Default (0 = No, 1 = Charged Off)')
plt.ylabel('Delinquency Risk Index (Log Scale)')
plt.show()
```


    
![png](output_108_0.png)
    


# MULTIVARIATE ANALYSIS

## 1. Interaction Between Loan Amount, Interest Rate, and DTI
We aim to identify how loan amount, interest rate, and debt-to-income ratio (DTI) affect the likelihood of default. This can help in determining high-risk loans.


```python
# Plot the interaction between loan_amnt, int_rate, and dti
plt.figure(figsize=(10, 6))
plt.hexbin(df['loan_amnt'], df['int_rate'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.title('Hexbin of Loan Amount vs Interest Rate')
plt.xlabel('Loan Amount')
plt.ylabel('Interest Rate')
plt.show()
```


    
![png](output_111_0.png)
    


## 2. Income and Employment Length vs Loan Default
Analyze how a combination of employment length and annual income predicts the likelihood of default. The assumption here is that shorter employment length and lower income levels might correlate with higher default rates.


```python
# Create a bivariate plot to explore the relationship between emp_length, annual_inc, and loan_status
plt.figure(figsize=(12, 6))
sns.violinplot(x='emp_length', y='annual_inc', hue='loan_status_binary', data=df, split=True, palette='Set3')
plt.title('Income vs Employment Length and Loan Status (Violin Plot)')
plt.xlabel('Employment Length (Years)')
plt.ylabel('Annual Income')
plt.legend(title='Loan Status', loc='upper right')
plt.show()
```


    
![png](output_113_0.png)
    


## 3. Loan Grade, Home Ownership, and DTI
Analyze how loan grades (assigned by the lending company), home ownership status, and debt-to-income ratio together affect the likelihood of default. This can highlight how credit risk assessments combine with personal circumstances.


```python
# Plot the relationship between grade, home_ownership, and dti with loan status
plt.figure(figsize=(12, 6))
sns.violinplot(x='grade', y='dti', hue='loan_status_binary', data=df, split=True, inner='quartile', palette='muted')
plt.title('DTI Distribution by Grade and Loan Status')
plt.xlabel('Loan Grade')
plt.ylabel('Debt-to-Income Ratio (DTI)')
plt.legend(title='Loan Status', loc='upper right')
plt.show()
```


    
![png](output_115_0.png)
    


## 4. Loan Amount, Term, and Default
Longer loan terms with higher loan amounts could be an indicator of risk due to potential long-term financial instability.


```python
# Loan Term vs Loan Amount with Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='term', y='loan_amnt', hue='loan_status_binary', data=df)
plt.title('Loan Term vs Loan Amount and Loan Status')
plt.xlabel('Loan Term (Months)')
plt.ylabel('Loan Amount')
plt.show()
```


    
![png](output_117_0.png)
    


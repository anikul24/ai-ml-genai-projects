CREATE DATABASE Credit_risk_prediction;

CREATE SCHEMA Credit_risk_prediction.credit_data;


use role accountadmin;

create Warehouse if not exists Credit_risk_WH
with warehouse_size = 'X-Small'
max_cluster_count = 1
min_cluster_count = 1
auto_suspend = 3
initially_suspended = TRUE
comment = "WH for credit_data";

create role if not exists Credit_risk_sysadmin;

SELECT CURRENT_USER();


GRANT ROLE CREDIT_RISK_SYSADMIN TO USER AKULKARNI24; 


grant usage on database Credit_risk_prediction to role Credit_risk_sysadmin;

grant usage on schema credit_data to role Credit_risk_sysadmin;


grant all privileges on all schemas in database  Credit_risk_prediction to role Credit_risk_sysadmin;
grant all privileges on future schemas in database  Credit_risk_prediction to role Credit_risk_sysadmin;

grant modify, monitor,usage,operate on warehouse Credit_risk_WH to Credit_risk_sysadmin;

grant ownership on all schemas in database  Credit_risk_prediction to role Credit_risk_sysadmin copy current grants;
grant ownership on all procedures in database  Credit_risk_prediction to role Credit_risk_sysadmin copy current grants;
grant ownership on all tables in database  Credit_risk_prediction to role Credit_risk_sysadmin copy current grants;
grant ownership on future tables in database  Credit_risk_prediction to role Credit_risk_sysadmin copy current grants;
grant ownership on all tasks in database  Credit_risk_prediction to role Credit_risk_sysadmin copy current grants;
grant ownership on future tasks  in database  Credit_risk_prediction to  role Credit_risk_sysadmin copy  current grants;

grant usage on warehouse Credit_risk_WH to Credit_risk_sysadmin;

use role Credit_risk_sysadmin;



CREATE OR REPLACE FILE FORMAT csv_format
TYPE = 'CSV'
FIELD_DELIMITER = ','
SKIP_HEADER = 1;


CREATE OR REPLACE FILE FORMAT csv_format
TYPE = 'CSV'
FIELD_DELIMITER = ','
NULL_IF = ('', 'NA')
EMPTY_FIELD_AS_NULL = TRUE
SKIP_HEADER = 1;



CREATE OR REPLACE STAGE credit_csv_stage
FILE_FORMAT = csv_format;

CREATE OR REPLACE FILE FORMAT csv_format
TYPE = 'CSV'
FIELD_DELIMITER = ','
SKIP_HEADER = 1;

describe stage credit_csv_stage


## Upload the files to the stage from local using snowSQL command line
PUT 'file://../Github/Credit Risk Prediction/notebook/data/cs-test.csv' @credit_csv_stage;


PUT 'file://../Github/Credit Risk Prediction/notebook/data/cs-training.csv' @credit_csv_stage;


list @credit_csv_stage


 SELECT CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA();


select count(1) from CS_TEST; --101503



train -- 150000
test ---101503


select count(1) from CS_TRAIN;  --150000

select * from CS_TRAIN limit 10;


CREATE TABLE CS_TEST (
    SRC_DLQ_TWO_YRS INTEGER ,
    RVLV_UTLZ_UNSEC_LINES FLOAT,
    AGE INTEGER,
    NUM_TIMES_PAST_DUE_IN_30_59_DAYS INTEGER,
    DEBT_RATIO FLOAT,
	MNTHLY_INCOME INTEGER,
	NUM_TIMES_OPEN_CRDT_LINES_LOANS INTEGER,
	NUM_TIMES_90_DAYS_LATE INTEGER,
	NUM_TIMES_REAL_ESTATE_LOAN_LINES INTEGER,
	NUM_TIMES_60_59_DAYS_PAST_DUE INTEGER,
	NUM_DEPEDENTS  INTEGER
);

CREATE TABLE CS_TRAIN (
    SRC_DLQ_TWO_YRS INTEGER ,
    RVLV_UTLZ_UNSEC_LINES FLOAT,
    AGE INTEGER,
    NUM_TIMES_PAST_DUE_IN_30_59_DAYS INTEGER,
    DEBT_RATIO FLOAT,
	MNTHLY_INCOME INTEGER,
	NUM_TIMES_OPEN_CRDT_LINES_LOANS INTEGER,
	NUM_TIMES_90_DAYS_LATE INTEGER,
	NUM_TIMES_REAL_ESTATE_LOAN_LINES INTEGER,
	NUM_TIMES_60_59_DAYS_PAST_DUE INTEGER,
	NUM_DEPEDENTS  INTEGER
);



CREATE TEMPORARY TABLE CS_TEST_TEMP (
    ROW_ID  INTEGER ,
    SRC_DLQ_TWO_YRS INTEGER ,
    RVLV_UTLZ_UNSEC_LINES FLOAT,
    AGE INTEGER,
    NUM_TIMES_PAST_DUE_IN_30_59_DAYS INTEGER,
    DEBT_RATIO FLOAT,
	MNTHLY_INCOME INTEGER,
	NUM_TIMES_OPEN_CRDT_LINES_LOANS INTEGER,
	NUM_TIMES_90_DAYS_LATE INTEGER,
	NUM_TIMES_REAL_ESTATE_LOAN_LINES INTEGER,
	NUM_TIMES_60_59_DAYS_PAST_DUE INTEGER,
	NUM_DEPEDENTS  INTEGER
);


COPY INTO CS_TEST ( SRC_DLQ_TWO_YRS  ,     RVLV_UTLZ_UNSEC_LINES ,     AGE ,     NUM_TIMES_PAST_DUE_IN_30_59_DAYS ,     DEBT_RATIO , 	MNTHLY_INCOME , 	NUM_TIMES_OPEN_CRDT_LINES_LOANS , 	NUM_TIMES_90_DAYS_LATE , 	NUM_TIMES_REAL_ESTATE_LOAN_LINES , 	NUM_TIMES_60_59_DAYS_PAST_DUE , 	NUM_DEPEDENTS )
FROM (
    SELECT
        $2,   
        $3,   
        $4,
        $5,
        $6,
        $7,
        $8,
        $9,
        $10,
        $11,
        $12
        
    FROM @credit_csv_stage/cs-test.csv.gz
)
FILE_FORMAT = (FORMAT_NAME = Credit_risk_prediction.credit_data.csv_format);


COPY INTO CS_TRAIN( SRC_DLQ_TWO_YRS  ,     RVLV_UTLZ_UNSEC_LINES ,     AGE ,     NUM_TIMES_PAST_DUE_IN_30_59_DAYS ,     DEBT_RATIO , 	MNTHLY_INCOME , 	NUM_TIMES_OPEN_CRDT_LINES_LOANS , 	NUM_TIMES_90_DAYS_LATE , 	NUM_TIMES_REAL_ESTATE_LOAN_LINES , 	NUM_TIMES_60_59_DAYS_PAST_DUE , 	NUM_DEPEDENTS )
FROM (
    SELECT
        $2,   
        $3,   
        $4,
        $5,
        $6,
        $7,
        $8,
        $9,
        $10,
        $11,
        $12
        
    FROM @credit_csv_stage/cs-training.csv.gz
)
FILE_FORMAT = (FORMAT_NAME = Credit_risk_prediction.credit_data.csv_format);


select * from Credit_risk_prediction.credit_data.cs_test 
where MNTHLY_INCOME is null
limit 5;
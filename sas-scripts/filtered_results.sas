PROC SQL;
CREATE TABLE WORK.query AS
SELECT spid , rl13dcondspanh , rl13dspkothlan , rl13dunspokeng , rl13dracehisp , cg13ratememry , wb13ageyofeel , ip13numyears , lf13doccpctgy , cg13dclkdraw , cg13dclkcoding , cg13dclkextract , cg13dclkcoded , r13dnhatssppb , r13dnhatsbasc , r13dorigsppb , ir13intlangpt2 FROM WORK.SPFILE WHERE (cg13dclkdraw=0 OR cg13dclkdraw=1 OR cg13dclkdraw=2 OR cg13dclkdraw=3 OR cg13dclkdraw=4 OR cg13dclkdraw=5) ORDER BY cg13dclkdraw DESCENDING;
RUN;
QUIT;

PROC DATASETS NOLIST NODETAILS;
CONTENTS DATA=WORK.query OUT=WORK.details;
RUN;

PROC PRINT DATA=WORK.details;
RUN;

proc export data=work.query
    outfile="/home/u64172862/sasuser.v94/Query_Output.xlsx"
    dbms=xlsx
    label
    replace;
run;
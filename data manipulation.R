rm(list=ls(all=TRUE))


### read college financial data with matched university ID
finance=read.csv("data/college_financials_with_id_higher_threshold.csv",header=T)

### exclude the ones 
exclude=which(finance$unit.id==-1)
finance=finance[-exclude,]


college=read.csv("data/colleges.csv")

### data include financial index
# financial_index=read.csv("all_dimred_linear.csv")

### use neural network financial index
financial_index=read.csv("all_dimred_nn.csv")

### match unit id to financial index data
finance_index=merge(finance,financial_index,by=c("inst_name","academic_year"))

finance_index2=finance_index[,c("inst_name","academic_year","unit.id","sector.x","eandr","total_faculty_all",
                                "index_1", "index_2", "index_3", "index_4", "index_5"
)]


colnames(finance_index2)[2]="YEAR"
colnames(finance_index2)[3]="UNITID"


combine=merge(college,finance_index2,by=c("UNITID","YEAR"))

dim(combine)  ## 32070 92


completion_rate=combine$C150_4
completion_rate[is.na(completion_rate)]=combine$C150_L4[is.na(combine$C150_4)]

combine=cbind(combine,completion_rate)

Score=combine[,c("UNITID","YEAR","INSTNM","STABBR","ADM_RATE","SATVR75","CONTROL",
                 "SATMT75", "eandr","completion_rate","AVGFACSAL","UGDS","total_faculty_all")]


### public
public_score=Score[which(Score$CONTROL==1),]
dim(public_score)  ## 12771    13
# 
Score2=na.omit(public_score)
dim(Score2)  ## 4760 13
# 
# 
Score3=Score2[-which(Score2$UGDS==0),]
# 
## faculty/student ratio
which(Score3$UGDS==0)
faculty_student_ratio=Score3$total_faculty_all/Score3$UGDS
faculty_student_ratio=faculty_student_ratio/max(faculty_student_ratio)
# 
range(faculty_student_ratio)  ## max=1.77
# 
# 
# 
# 
### SAT score
SAT=(Score3$SATVR75/800+Score3$SATMT75/800)/2
range(SAT)
### average faculty salary
faculty_salary=Score3$AVGFACSAL
faculty_salary=faculty_salary/max(faculty_salary)
range(faculty_salary)
## financial resources per student (education expenditures per student)
education_expense=Score3$eandr/Score3$UGDS
education_expense=education_expense/max(education_expense)
# 
range(education_expense)
# 
classsize=Score3$UGDS
classsize=classsize/max(classsize)
range(classsize)
# 
admission_rate=Score3$ADM_RATE
range(admission_rate)
# 
rankscore=0.35*Score3$completion_rate+0.3*(0.45*faculty_salary+0.05*faculty_student_ratio+0.5*classsize)+0.2*(0.15*admission_rate+0.85*SAT)+0.15*education_expense
# 
# 
Score4=cbind(Score3,faculty_salary,faculty_student_ratio,classsize,admission_rate,SAT,education_expense,rankscore)
# 
range(rankscore)
which.max(rankscore)
# 
write.csv(Score4,file="ranking_score_public.csv",row.names = F)

Score4=read.csv("ranking_score.csv")

public_nn=merge(combine,Score4,by=c("UNITID","YEAR"))

dim(public_nn)  ## 2671 110

write.csv(public_nn,file="public_nn_merged171119.csv",row.names=F)

#### create dummy variable for categorical variable
install.packages("dummies")
library(dummies)


public.cate=public_nn[,c("YEAR","STABBR.x","PREDDEG","HIGHDEG",
                         "MAIN","sector.x")]

colnames(public.cate)[2]="state"
colnames(public.cate)[6]="sector"

## x"CONTROL",


public.cate$YEAR=as.factor(public.cate$YEAR)

public.cate$PREDDEG=as.factor(public.cate$PREDDEG)

public.cate$HIGHDEG=as.factor(public.cate$HIGHDEG)


public.cate$sector=as.factor(public.cate$sector)

public.cate2=dummy.data.frame(public.cate,sep="_")

apply(public.cate2,2,sum)==0

publicdata_nn=cbind(public_nn,public.cate2)

# write.csv(publicdata_nn,file="public_cate_merged_nn.csv",row.names=F)



names=c("UNITID","YEAR","INSTNM.x","STABBR.x",
        "CUML_DEBT_N",
        "CUML_DEBT_P90" , "CUML_DEBT_P75" ,"CUML_DEBT_P25",                
        "CUML_DEBT_P10" ,
        "PCIP01","PCIP03",                        
        "PCIP04", "PCIP05", "PCIP09",                        
        "PCIP10", "PCIP11", "PCIP12",                        
        "PCIP13", "PCIP14", "PCIP15",                        
        "PCIP16", "PCIP19", "PCIP22",                        
        "PCIP23" , "PCIP24", "PCIP25",                        
        "PCIP26", "PCIP27", "PCIP29",                        
        "PCIP30", "PCIP31", "PCIP38",                        
        "PCIP39", "PCIP40", "PCIP41",                        
        "PCIP42", "PCIP43", "PCIP44",                        
        "PCIP45" , "PCIP46", "PCIP47",                        
        "PCIP48", "PCIP49" , "PCIP50",                        
        "PCIP51", "PCIP52", "PCIP54",
        "TUITIONFEE_IN","TUITIONFEE_OUT",
        colnames(public.cate2),"index_1","index_2","index_3","index_4","index_5","rankscore")


use=publicdata_nn[,names]
dim(use)


# write.csv(use,file="public_cate_merged_nn.csv",row.names=F)

write.csv(use,file="public_cate_merged_nn171119.csv",row.names=F)


### private nonprofit
private_nonprofit_score=Score[which(Score$CONTROL==2),]
dim(private_nonprofit_score)  ## 12771    13

Score2=na.omit(private_nonprofit_score)
dim(Score2)  ## 4760 13


Score3=Score2[-which(Score2$UGDS==0),]

## faculty/student ratio
which(Score3$UGDS==0)
faculty_student_ratio=Score3$total_faculty_all/Score3$UGDS
faculty_student_ratio=faculty_student_ratio/max(faculty_student_ratio)

range(faculty_student_ratio)  ## max=1.77




### SAT score
SAT=(Score3$SATVR75/800+Score3$SATMT75/800)/2
range(SAT)
### average faculty salary
faculty_salary=Score3$AVGFACSAL
faculty_salary=faculty_salary/max(faculty_salary)
range(faculty_salary)
## financial resources per student (education expenditures per student)
education_expense=Score3$eandr/Score3$UGDS
education_expense=education_expense/max(education_expense)

range(education_expense)

classsize=Score3$UGDS
classsize=classsize/max(classsize)
range(classsize)

admission_rate=Score3$ADM_RATE
range(admission_rate)

rankscore=0.35*Score3$completion_rate+0.3*(0.45*faculty_salary+0.05*faculty_student_ratio+0.5*classsize)+0.2*(0.15*admission_rate+0.85*SAT)+0.15*education_expense


Score4=cbind(Score3,faculty_salary,faculty_student_ratio,classsize,admission_rate,SAT,education_expense,rankscore)

range(rankscore)
which.max(rankscore)

write.csv(Score4,file="ranking_score_privatenonprofit.csv",row.names = F)

Score4=read.csv("ranking_score_privatenonprofit.csv")

private_nonprofit=merge(combine,Score4,by=c("UNITID","YEAR"))

dim(private_nonprofit)  ## 6077 111

write.csv(private_nonprofit,file="private_nonprofit_merged.csv",row.names=F)

write.csv(private_nonprofit,file="private_nonprofit_merged_nn.csv",row.names=F)

write.csv(private_nonprofit,file="private_nonprofit_merged_nn171119.csv",row.names=F)

#### create dummy variable for categorical variable
install.packages("dummies")
library(dummies)


private.cate=private_nonprofit[,c("YEAR","STABBR.x","PREDDEG","HIGHDEG",
                                  "MAIN","sector.x")]

colnames(private.cate)[2]="state"
colnames(private.cate)[6]="sector"

## x"CONTROL",


private.cate$YEAR=as.factor(private.cate$YEAR)

private.cate$PREDDEG=as.factor(private.cate$PREDDEG)

private.cate$HIGHDEG=as.factor(private.cate$HIGHDEG)


private.cate$sector=as.factor(private.cate$sector)

private.cate2=dummy.data.frame(private.cate,sep="_")

apply(private.cate2,2,sum)==0

privatedata=cbind(private_nonprofit,private.cate2)

write.csv(privatedata,file="private_cate_merged.csv",row.names=F)



names=c("UNITID","YEAR","INSTNM.x","STABBR.x",
        "CUML_DEBT_N",
        "CUML_DEBT_P90" , "CUML_DEBT_P75" ,"CUML_DEBT_P25",                
        "CUML_DEBT_P10" ,
        "PCIP01","PCIP03",                        
        "PCIP04", "PCIP05", "PCIP09",                        
        "PCIP10", "PCIP11", "PCIP12",                        
        "PCIP13", "PCIP14", "PCIP15",                        
        "PCIP16", "PCIP19", "PCIP22",                        
        "PCIP23" , "PCIP24", "PCIP25",                        
        "PCIP26", "PCIP27", "PCIP29",                        
        "PCIP30", "PCIP31", "PCIP38",                        
        "PCIP39", "PCIP40", "PCIP41",                        
        "PCIP42", "PCIP43", "PCIP44",                        
        "PCIP45" , "PCIP46", "PCIP47",                        
        "PCIP48", "PCIP49" , "PCIP50",                        
        "PCIP51", "PCIP52", "PCIP54",
        "TUITIONFEE_IN","TUITIONFEE_OUT",
        colnames(private.cate2),"index_1","index_2","index_3","index_4","index_5","rankscore")


use=privatedata[,names]
dim(use)  ## 6077 127


write.csv(use,file="private_cate_merged_nn171119.csv",row.names = F)


write.csv(use,file="private_cate_merged_nn.csv",row.names = F)




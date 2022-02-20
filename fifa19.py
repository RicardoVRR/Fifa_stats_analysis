# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os                         #sistema operativo
import pandas as pd               #geestionar datframes. Soporte principal para estructurar datos
import numpy as np                #numeric python vectores
import matplotlib.pyplot as plt 
import scipy.stats as stats #for statistical tests
import seaborn as sns
from pandas.api.types import CategoricalDtype
from scipy.stats.stats import chisquare
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer


os.chdir('/Users/RicardoVReig/Desktop/apuntesdata/PEP12-11/data_fifa')

fifa = pd.read_csv ('fifaplayers.csv')
pd.set_option('display.max_columns', 100)
pd.set_option("max_rows", 100)

fifa.dtypes

"""
variable target: wage: cuantitativa
variables predictoras: 
    Overall:        cuantitativa, 
    Prefered foot:  nominal, 
    Value:          cuantitativa, 
    Position:       nominal, 
    Height:         cuantitativa, 
    Finishing:      cuantitativa, 
    Value:          cuantitativa, 
    Age:            cuantitativa,
    
H1: a mayor overall (calidad), mayor salario
H2: si es jugador es left, mayor salario
H3: a mayor market value, mayor salario
H4: a mayor edad, menor salario
H5: si el jugador es delantero, mayor salario (dividir en 3 grupos:defensa, medio y delantero)
H6: a mayor height, mayor salario
H7: a mayor finishing, mayor salario
H8: a mayor contract valid until, menor salario
"""

print(fifa.shape)

#cleaning dependent variable
fifa['Wage']=fifa.Wage.replace(0,np.nan)
fifa['Wage2'] = fifa['Wage'].str.replace('€','').astype(object)
fifa['Wage3'] = fifa['Wage2'].str.replace('K','').astype(int)
fifa['Wage3']=fifa.Wage3.replace(0,np.nan)

fifa.Wage3.describe()
plt.hist(fifa.Wage3)

#-----------------------------h1 a mayor overall (calidad), mayor salario

fifa.Overall.describe()
plt.hist(fifa.Overall)
plt.hist(fifa.Overall.dropna())
y=fifa.Wage3
x1=fifa.Overall.dropna()
x1.describe()
from scipy.stats.stats import pearsonr
res=pearsonr(x1, y)
print (res)
#me da un error de "array must not contain infs or nans, compruebo:
np.isnan(x1).any()
np.isnan(y).any() #aquí está

y=fifa.Wage3.dropna()

pearsonr(x1,y) #error: ValueError: x and y must have the same length.

x1.shape #18207
y.shape #17966
#creo una nueva tabla para elminar los nans en las dos columnas
fifa2= pd.DataFrame ({'Overall': fifa.Overall, 'Wage':fifa.Wage3})
fifa3=fifa2.dropna()

#plot variable
fifa3.Overall.describe()
plt.hist(fifa3.Overall,edgecolor='black') #pongo el color negro al eje x que he denominado antes
plt.xticks(np.arange(30, 109, step=10)) #los valores y trampos para el eje x
plt.title('Figure. Overall quality \n players')
plt.ylabel('Frecuency')
plt.xlabel('Overall quality')
plt.show()

x11=fifa3.Overall
x11.shape
y1=fifa3.Wage
y1.shape

res=pearsonr(x11,y1)
r1=res[0]
p_val1=res[1]
n1=y1.shape

#CORRELATION
plt.figure(figsize=(10, 5))
plt.scatter (fifa3.Overall, fifa3.Wage, s=20, facecolors='none', edgecolors='C1')
#x dependiente, y indep. c separo por colores a traves de una variable(años?)
plt.xticks(np.arange(40, 110, step=10))
plt.yticks(np.arange(0, 600, step=50))
plt.title('Figure 9. Player Wages per Overall quality''\n' )
plt.ylabel('Wage in thousands (€)')
plt.xlabel('Player quality score')
props = dict(boxstyle='round', facecolor='white', lw=0.1)
textstr = '$\mathrm{r}=%.2f$\n$\mathrm{P.Val:}=%.3f$\n$\mathrm{n}=%.0f$'%(r1, p_val1, n1)
plt.text(50,450,textstr, bbox=props) 
plt.savefig('wage_overall.eps')
plt.show()


#As pvalue < 0, we reject h0 with confidence >99,9%. 
#There is linear association between wage and quality (overall)


#------------------H2: si es jugador es left, mayor salario


fifa.dtypes
fifa4= pd.DataFrame ({'Preferred Foot': fifa['Preferred Foot'], 'Wage':fifa.Wage3})
fifa5=fifa4.dropna()
pfoot = fifa5.groupby(['Preferred Foot']).size()
#otra manera para agrupar;
pfoot2=pd.crosstab(index=fifa5['Preferred Foot'], columns='Wage3')
# %
n=pfoot.sum()
pfoot2 = (pfoot/n)*100
pfoot3=round(pfoot2,1)
print(pfoot3)
#plot
bar_list = ['Left', 'Right']
plt.bar(bar_list, pfoot3)
plt.title ('Figure 1. Percentage of left and right players')
plt.ylabel ('Percentage')
props = dict(boxstyle ='round', facecolor ='white', lw = 5)
textstr = 'mathrm{n}=%.0f$'%(n)
plt.text (0.01,70, 'n=18159' , bbox = props)

fifa5.Wage
plt.hist(fifa5.Wage)

#descriptive comparison of mean wage depending on foot
fifa5.groupby('Preferred Foot').Wage.mean()
#agrupar en dos variables los jugadores left y right, solo con su WAGE
wage_left = fifa5.loc[fifa5['Preferred Foot']=='Left', 'Wage']
wage_right = fifa5.loc[fifa5['Preferred Foot']=='Right', 'Wage']

res2=stats.ttest_ind(wage_left , wage_right , equal_var = False)
F2=res2[0]
P_val2=res2[1]
n2=fifa5.shape

#pvalue= 0.063
#We fail to reject h0, we cannot say there is a relation between foot and wage
#Average wage does not differ in left foot and right foot

#plot:
plt.figure(figsize=(5,5))
#EN LA X PONEMOS LA VARIABLE NOMINAL Y EN LA Y LA VARIABLE CUANTITATIVA
ax = sns.pointplot(x="Preferred Foot", y="Wage", data=fifa5,ci=95, join=0) 
plt.yticks(np.arange(4, 16, step=2))
plt.ylim(4,16)
plt.axhline(y=fifa5.Wage.mean(), #aquí ponemos una linea con la media de las ventas
            linewidth=1,
            linestyle= 'dashed',
            color="green")
props = dict(boxstyle='round', facecolor='white', lw=0.5)
plt.text(0.75,13,'Mean:9.88K/Month''\n''n=17918' '\n' 'F:1.854' '\n' 'Pval.:0.063', bbox=props) 
plt.xlabel('Preferred Foot')
plt.title('Figure 6. Average wage by Foot preference.''\n')
plt.savefig('footcomparison.eps')
plt.savefig('footcomparison.jpg')
fifa5.describe()
plt.show()




#------------H3: a mayor market value, mayor salario


fifa.Value.describe()

#clean de variable to a int
fifa['Value']=fifa.Value.replace('€0',np.nan)
#aquí se intenta eliminar el . y el siguiente caracter que tenga. si NO HABIA . ME REPITE EL ULTIMO CARACTER
#Necesito saber el ultimo caractar M o K para tener el valor de mercado en miles o millones
#revisar:
fifa['Value2'] = fifa['Value'].str.split('.').str[0]+fifa.Value.str[-1]

fifa['Value3'] = fifa['Value2'].str.replace('€','').astype(object)
fifa['Value4'] = fifa['Value3'].str.replace('MM', '000').astype(object)
fifa['Value5'] = fifa['Value4'].str.replace('M','000').astype(object)
fifa['Value6'] = fifa['Value5'].str.replace('K','').astype(float)
#forma mejorada para hacerlo:
fifa['Value3'] = fifa['Value2'].str.replace('€','') \
    .str.replace('MM', '000') \
    .str.replace('M', '000') \
    .str.replace('K','').astype(float)

fifa.Value3.shape
fifa.Wage3.shape
np.isnan(fifa.Value3).any() #true

fifa.dtypes
#Need same shape to perform test:
fifa6= pd.DataFrame ({'Value': fifa.Value3, 'Wage':fifa.Wage3})
fifa6.shape
fifa7=fifa6.dropna()
fifa7.shape

fifa7.Value.describe()
plt.hist(fifa7.Value,edgecolor='black') #pongo el color negro al eje x que he denominado antes
plt.xticks(np.arange(0, 120000, step=20000)) #los valores y trampos para el eje x
plt.title('Figure. Players Value \n ')
plt.ylabel('Frecuency')
plt.xlabel('Value in thousands of €')
plt.show()

#test
r3, p_val3 =pearsonr(fifa7.Value, fifa7.Wage)
print(r3, p_val3)
#As pvalue < 0.0, we reject h0 with confidence >99,9%. 
#There is linear associaciotn between Wage and Value
n = len (fifa7.Wage)
n

print( 'r:', round(r3,3), 'P.Val:', round(p_val3,3), 'n:', n)

#CORRELATION
plt.figure(figsize=(10, 5))
plt.scatter (fifa7.Value, fifa7.Wage, s=20, facecolors='none', edgecolors='C0')
#x dependiente, y indep. c separo por colores a traves de una variable(años?)
plt.xticks(np.arange(0, 120001, step=20000))
plt.yticks(np.arange(0, 600, step=50))
plt.title('Figure 9. Player Wages per Market Value''\n' )
plt.ylabel('Wages')
plt.xlabel('Value')
props = dict(boxstyle='round', facecolor='white', lw=0.1)
textstr = '$\mathrm{r}=%.2f$\n$\mathrm{P.Val:}=%.3f$\n$\mathrm{n}=%.0f$'%(r3, p_val3, n)
plt.text(3,450,textstr, bbox=props) 
plt.savefig('wage_value.eps')
plt.show()
#above in the graph we can see the positive correlation.



#---------------- H4: a mayor edad, menor salario
reset -f

fifa8= pd.DataFrame ({'Age': fifa.Age, 'Wage':fifa.Wage3, 'Foot':fifa['Preferred Foot']})
fifa9=fifa8.dropna()


res4=fifa9.Wage.describe()
n=res4[0]
mean=res4[1]
sd=res4[2]
fifa9.Wage.describe()

plt.hist(fifa9.Wage)
#recoding dependiente:
my_categories=["Low Wages", "Average Wages", "High Wages"]
my_wage_type = CategoricalDtype(categories=my_categories, ordered=True)

fifa9.loc[(fifa9['Wage']<(mean)) ,"Wage_cat"]= "Low Wages"
fifa9.loc[((fifa9['Wage']>=(mean)) & (fifa9['Wage']<(mean+sd))) ,"Wage_cat"]= "Average Wages"
fifa9.loc[(fifa9['Wage']>=(mean+sd)) ,"Wage_cat"]= "High Wages"
fifa9.dtypes

fifa9["Wage_cat"] = fifa9.Wage_cat.astype(my_wage_type)
fifa9.info()

plt.scatter(fifa9.Wage, fifa9.Wage_cat, s=1)
#QCOK
#plot
wagebar = fifa9.groupby(['Wage_cat']).size()
n4=wagebar.sum()
wagebar2=(wagebar/n)*100
wagebar3=wagebar2.round(1)
print(wagebar3)

plt.bar(my_categories, wagebar3, edgecolor = 'black')
plt.ylabel('Percentage')
plt.title('Distribution of wages per player')
plt.show()

#recoding indep

res5=fifa9.Age.describe()
n5=res5[0]
mean5=res5[1]
sd5=res5[2]
res5

plt.hist(fifa9.Age)

my_categories2=['Rookies', 'Early 20s', 'Late 20s', 'Seniors']
my_age_types= CategoricalDtype(categories=my_categories2, ordered=True)

fifa9.loc[((fifa9['Age']<=20)), 'Age_cat']= 'Rookies'
fifa9.loc[((fifa9['Age']>(20)) & (fifa9['Age']<=(25))) ,"Age_cat"]= "Early 20s"
fifa9.loc[((fifa9['Age']>(25)) & (fifa9['Age']<=(30))) ,"Age_cat"]= "Late 20s"
fifa9.loc[(fifa9['Age']>(30)) ,"Age_cat"]= "Seniors"

fifa9["Age_cat"] = fifa9.Age_cat.astype(my_age_types)

fifa9.info()

plt.scatter(fifa9.Age, fifa9.Age_cat, s=1)
#QCOK

#plot:
agebar = fifa9.groupby(['Age_cat']).size()
n4=agebar.sum()
agebar2=(agebar/n)*100
agebar3=agebar2.round(1)
print(agebar3)

plt.bar(my_categories2, agebar3, edgecolor = 'black')
plt.ylabel('Percentage')
plt.title('Distribution of age per player')
plt.show()

#(Cross tabulation of DV by IV)
my_ct=pd.crosstab(fifa9.Wage_cat, fifa9.Age_cat, normalize='columns', margins=True)*100
my_ct=round(my_ct, 1)
my_ct

#test
res6=stats.chi2_contingency(my_ct)
p_val6=res6[1]
r6=res6[0]
n6=fifa9.Wage.count()

#As pvalue < 0.0, we reject h0 with confidence >99,9%. 
#Percentage of Wages with low/average/high wage differs in 
#different range of ages.
plt.figure(figsize=(50, 50))
my_ct2 = my_ct.transpose()
my_ct2.plot(kind="bar", edgecolor= "black", colormap="Greens")
plt.ylim(0,120)
plt.xticks(rotation='horizontal')
props = dict(boxstyle="round", facecolor="white", lw=0.5)
plt.legend(['Low Wages', 'Average Wages', 'High Wages'])
plt.text(-0.4, 100, "Chi2: 30.77""\n""n:17918""\n""Pval:0.00", bbox=props)
plt.xlabel("Income")
plt.ylabel('Distribution of players')
plt.title("Figure. Players Income by range of age")
plt.savefig('Wages_age.eps')
plt.savefig('Wages_age.jpg')

#correlation

plt.figure(figsize=(7, 5))
colors = {'Left':'orange', 'Right':'blue'}
plt.scatter (fifa9.Age, fifa9.Wage, s=20, facecolors='none', c=fifa9['Foot'].map(colors))
#x dependiente, y indep. c separo por colores a traves de una variable(años?)
plt.xticks(np.arange(15, 47, step=2))
plt.yticks(np.arange(0, 600, step=46))
plt.title('Figure 9. Player Wages per Market Value''\n' )
plt.ylabel('Wages')
plt.xlabel('Age')
props = dict(boxstyle='round', facecolor='white', lw=0.1)
plt.legend(['Left foot', 'Right foot']) #no sé como sacar leyenda de los dos
textstr = '$\mathrm{r}=%.2f$\n$\mathrm{P.Val:}=%.3f$\n$\mathrm{n}=%.0f$'%(r6, p_val6, n6)
plt.text(15,450,textstr, bbox=props) 
plt.savefig('wage_age2.eps')
plt.savefig('wage_age2.jpg')
plt.show()

#we can see a positive correlation btween wages and age up to 27-28 years
#at this point it turns the other way being a negative correlation


#---------REGRESSION
fifa10= pd.DataFrame ({'Age': fifa9.Age, 'Wage':fifa.Wage3, 'Foot':fifa5['Preferred Foot'], 'Overall':fifa3.Overall, 'Value':fifa7.Value})
fifa10=fifa10.dropna() #tengo unas pocas menos filas que cuando he hecho la regresión

model1 = ols('Wage ~ Overall', data=fifa10).fit()
model1.summary2()

model2= ols('Wage ~ Overall + Foot', data=fifa10).fit()
model2.summary2()

model3= ols('Wage ~ Overall + Foot + Value', data=fifa10).fit()
model3.summary2()

model4= ols('Wage ~ Overall + Foot + Value + Age', data=fifa10).fit()
model4.summary2()

'''
#comments:
    
   In the end R2 shows a value of 0.741, meaning that the model explains with accuracy the variance in the dependent variables
    with the iv chosen. Being Value the most representative variable.
    To achieve a further analysis, a separation of age in two groups should be made. Since up to 27 years of age we can see a positive
    correlation between iv and dv, but from this point onwards the slope becomes flatter, even downward
    
'''
    




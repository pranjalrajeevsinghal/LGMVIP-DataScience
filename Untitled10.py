#!/usr/bin/env python
# coding: utf-8

# <h3> Pranjal Singhal </h3>
# <h4> Exploratory Data Analysis on Dataset - Terrorism </h4>

# <h4> Loading our libraries </h4>

# In[112]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3> Loading Our Data </h3>
# 

# In[113]:


data = pd.read_csv("globalterrorismdb_0718dist.csv", encoding ="latin1")
df=pd.DataFrame(data)
pd.set_option('display.max_rows',None)


# In[114]:


data.head()


# In[115]:


df.shape


# In[116]:


df.info()


# In[117]:


df.describe()


# In[118]:


df.columns


# In[119]:


for i in df.columns:
  print(i, end= " , ")


# In[120]:


df=df[['iyear','imonth','iday','country_txt','region_txt','provstate','city','latitude','longitude','location','summary','attacktype1_txt','targtype1_txt','gname','motive','weaptype1_txt','nkill','nwound','addnotes']]
df.head()


# In[121]:


df.rename(columns={"iyear":"Year","imonth":"Month","iday":"Day","country_txt":"Country","provstate":"Province/State","city":"City","region_txt":"Region","latitude":"Latitude","longitude":"Longitude","location":"Location","summary":"Summary","attacktype1_txt":"Attack Type","targtype1_txt":"Target Type","gname":"Group Name","motive":"Motive","weaptype1_txt":"Weapon Type","nkill":"Killed","nwound":"Wounded","addnotes":"Add Notes"}, inplace=True)
df.head()


# In[122]:


df.info()


# In[123]:


df.isnull().sum()


# In[124]:


df["Killed"]=df["Killed"].fillna(0)
df["Wounded"]=df["Wounded"].fillna(0)
df["Casualty"]=df["Killed"]+df["Wounded"]


# In[125]:


df.describe()


# In[126]:


df.info()


# In[127]:


df.head()


# **Observations**
# 1. The data consists of terrorist activities ranging from: 1970 to 2017
# 2. maximum people killed in an event were 1570
# 3. Maximum number of people wounded in an event were: 8191
# 4. Maximum number of casualities happened in an event: 9574

# <h3> Visualising the Data </h3>

# 1. <h5> Year of attacks </h5>

# 1. Number of attacks in each year

# In[128]:


attacks=df["Year"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Year","Year":"Attacks"}).set_index("Year")
attacks.head()                                                                                                   


# In[129]:


attacks.plot(kind="bar",color="cornflowerblue",figsize=(15,6),fontsize=13)
plt.title("Timeline of Attacks",fontsize=15)
plt.xlabel("Years",fontsize=15)
plt.ylabel("Number of Attacks",fontsize=15)
plt.show()


# (i).   Most number of attacks(16903) in 2014.
# <br>
# (ii). Least number of attacks(471) in 1971

# 1. Total Casualties (Killed + Wounded) in each Year

# In[130]:


yc=df[["Year","Casualty"]].groupby("Year").sum()
yc.head()


# In[131]:


yc.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Year wise Casualties",fontsize=13)
plt.xlabel("Years",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# 1. Killed in each year

# In[132]:


yk=df[["Year","Killed"]].groupby("Year").sum()
yk.head()


# 1. Wounded in each Region

# In[133]:


yw=df[["Year","Wounded"]].groupby("Year").sum()
yw.head()


# In[134]:


fig=plt.figure()
ax0=fig.add_subplot(2,1,1)
ax1=fig.add_subplot(2,1,2)

#Killed
yk.plot(kind="bar",color="cornflowerblue",figsize=(15,15),ax=ax0)
ax0.set_title("People Killed in each Year")
ax0.set_xlabel("Years")
ax0.set_ylabel("Number of People Killed") 

#Wounded
yw.plot(kind="bar",color="cornflowerblue",figsize=(15,15),ax=ax1)
ax1.set_title("People Wounded in each Year")
ax1.set_xlabel("Years")
ax1.set_ylabel("Number of People Wounded")

plt.show()


# <h3> 2. Region Wise attacks </h3>
#     1. Distribution of Terrorist Attacks over Regions from 1970-2017

# In[135]:


reg=pd.crosstab(df.Year,df.Region)
reg.head()


# In[136]:


reg.plot(kind="area", stacked=False, alpha=0.5,figsize=(20,10))
plt.title("Region wise attacks",fontsize=20)
plt.xlabel("Years",fontsize=20)
plt.ylabel("Number of Attacks",fontsize=20)
plt.show()


# In[137]:


regt=reg.transpose()
regt["Total"]=regt.sum(axis=1)
ra=regt["Total"].sort_values(ascending=False)
ra


# In[138]:


ra.plot(kind="bar",figsize=(15,6))
plt.title("Total Number of Attacks in each Region from 1970-2017")
plt.xlabel("Region")
plt.ylabel("Number of Attacks")
plt.show()


# 1. Total Casualties (Killed + Wounded) in each Region

# In[139]:


rc=df[["Region","Casualty"]].groupby("Region").sum().sort_values(by="Casualty",ascending=False)
rc


# rc.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
# plt.title("Region wise Casualties",fontsize=13)
# plt.xlabel("Regions",fontsize=13)
# plt.xticks(fontsize=12)
# plt.ylabel("Number of Casualties",fontsize=13)
# plt.show()

# 1. Killed in each region

# In[140]:


rk=df[["Region","Killed"]].groupby("Region").sum().sort_values(by="Killed",ascending=False)
rk


# 1. Wounded in ech region

# In[141]:


rw=df[["Region","Wounded"]].groupby("Region").sum().sort_values(by="Wounded",ascending=False)
rw


# In[142]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
rk.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Region")
ax0.set_xlabel("Regions")
ax0.set_ylabel("Number of People Killed")

#Wounded
rw.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Region")
ax1.set_xlabel("Regions")
ax1.set_ylabel("Number of People Wounded")

plt.show()


# <h3> 3. Country wise Attacks - Top 10 </h3>

# 1. Number of Attacks in each Country

# In[143]:


ct=df["Country"].value_counts().head(10)
ct


# In[144]:


ct.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Country wise Attacks",fontsize=13)
plt.xlabel("Countries",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()


# 1. Total Casualties (Killed + Wounded) in each Country

# In[145]:


cnc=df[["Country","Casualty"]].groupby("Country").sum().sort_values(by="Casualty",ascending=False)
cnc.head(10)


# In[146]:


cnc[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Country wie Casualties",fontsize=13)
plt.xlabel("Countries",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# 1. Killed in each Country

# In[147]:


cnk=df[["Country","Killed"]].groupby("Country").sum().sort_values(by="Killed",ascending=False)
cnk.head(10)


# 1. Wounded in each country

# In[148]:


cnw=df[["Country","Wounded"]].groupby("Country").sum().sort_values(by="Wounded",ascending=False)
cnw.head(10)


# In[149]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
cnk[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Country")
ax0.set_xlabel("Countries")
ax0.set_ylabel("Number of People Killed")

#Wounded
cnw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Country")
ax1.set_xlabel("Countries")
ax1.set_ylabel("Number of People Wounded")

plt.show()


# <h3> 3. City wise Attacks - Top 10 </h3>
# <br>
# 1.Number of Attacks in each city

# In[150]:


city=df["City"].value_counts()[1:11]
city


# In[151]:


city.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("City wise Attacks",fontsize=13)
plt.xlabel("Cities",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()


# 1. Total Casualties (Killed + Wounded) in each City

# In[152]:


cc=df[["City","Casualty"]].groupby("City").sum().sort_values(by="Casualty",ascending=False).drop("Unknown")
cc.head(10)


# In[153]:


cc[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("City wise Casualties",fontsize=13)
plt.xlabel("Cities",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# 1, Killed in each City

# In[154]:


ck=df[["City","Killed"]].groupby("City").sum().sort_values(by="Killed",ascending=False).drop("Unknown")
ck.head(10)


# 1. Wounded in each City

# In[155]:


cw=df[["City","Wounded"]].groupby("City").sum().sort_values(by="Wounded",ascending=False).drop("Unknown")
cw.head(10)


# In[156]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
ck[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each City")
ax0.set_xlabel("Cities")
ax0.set_ylabel("Number of People Killed")

#Wounded
cw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each City")
ax1.set_xlabel("Cities")
ax1.set_ylabel("Number of People Wounded")

plt.show()


# <h3> 5. Terrorist Group wise Attacks - Top 10</h3>
# <br>
# 1. Number of Attacks by each Group

# In[157]:


grp=df["Group Name"].value_counts()[1:10]
grp


# In[158]:


grp.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Group wise Attacks",fontsize=13)
plt.xlabel("Terrorist Groups",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()


# 1. Total Casualties(Killed + Wounded) by each Group
# 

# In[159]:


gc=df[["Group Name","Casualty"]].groupby("Group Name").sum().sort_values(by="Casualty",ascending=False).drop("Unknown")
gc.head(10)


# In[160]:


gc.head(10).plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties by each Group",fontsize=13)
plt.xlabel("Terrorist Groups",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# 1. Killed by each Group

# In[161]:


gk=df[["Group Name","Killed"]].groupby("Group Name").sum().sort_values(by="Killed",ascending=False).drop("Unknown")
gk.head(10)


# 1. Wounded by each Group

# In[162]:


gw=df[["Group Name","Wounded"]].groupby("Group Name").sum().sort_values(by="Wounded",ascending=False).drop("Unknown")
gw.head(10)


# In[163]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
gk[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed by each Group")
ax0.set_xlabel("Terrorist Groups")
ax0.set_ylabel("Number of people Killed")

#Wounded
gw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded by each Group")
ax1.set_xlabel("Terrorist Groups")
ax1.set_ylabel("Number of people Wounded")
plt.show()


# <h3> 6. Attack Type wise Attacks </h3>
# <br>
# 1. Number of Attacks by each Attack Type

# In[164]:


at=df["Attack Type"].value_counts()
at


# In[165]:


at.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Types of Attacks",fontsize=13)
plt.xlabel("Attack Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()


# 1. Total Casualties (Killed + Wounded) by each Attack Type

# In[166]:


ac=df[["Attack Type","Casualty"]].groupby("Attack Type").sum().sort_values(by="Casualty",ascending=False)
ac


# In[167]:


ac.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties in each Attack",fontsize=13)
plt.xlabel("Attack Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# 1. Killed by each Attack Type

# In[168]:


ak=df[["Attack Type","Killed"]].groupby("Attack Type").sum().sort_values(by="Killed",ascending=False)
ak


# In[169]:


aw=df[["Attack Type","Wounded"]].groupby("Attack Type").sum().sort_values(by="Wounded",ascending=False)
aw


# In[170]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
ak.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Attack Type")
ax0.set_xlabel("Attack Types")
ax0.set_ylabel("Number of people Killed")

#Wounded
aw.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Attack Type")
ax1.set_xlabel("Attack Types")
ax1.set_ylabel("Number of people Wounded")
plt.show()


# <h3> 7. Target Type wise Attacks </h3>
# <br>
# 1. Number of Attacks over each Target Type

# In[171]:


ta=df["Target Type"].value_counts()
ta


# In[172]:


ta.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Types of Targets",fontsize=13)
plt.xlabel("Target Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()


# In[173]:


tc=df[["Target Type","Casualty"]].groupby("Target Type").sum().sort_values(by="Casualty",ascending=False)
tc


# In[174]:


tc.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties in each Target Attack",fontsize=13)
plt.xlabel("Target Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()


# In[175]:


tk=df[["Target Type","Killed"]].groupby("Target Type").sum().sort_values(by="Killed",ascending=False)
tk


# In[176]:


tw=df[["Target Type","Wounded"]].groupby("Target Type").sum().sort_values(by="Wounded",ascending=False)
tw


# In[177]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Killed
tk.plot(kind="bar",color="cornflowerblue",figsize=(17,6),ax=ax0)
ax0.set_title("People Killed in each Target Attack")
ax0.set_xlabel("Target Types")
ax0.set_ylabel("Number of people Killed")

#Wounded
tw.plot(kind="bar",color="cornflowerblue",figsize=(17,6),ax=ax1)
ax1.set_title("People Wounded in each Target Attack")
ax1.set_xlabel("Target Types")
ax1.set_ylabel("Number of people Wounded")
plt.show()


# <h3> Observations </h3>

# 1. Year wise Attacks :
# <br>
#   (i) Attacks
#   <br>
#     (a) Most number of attacks: 16903 in 2014
#     <br>
#     (b) Least number of attacks: 471 in 1971
#     <br>
#   (ii) Casualties
#   <br>
#     (a) Most number of casualties: 85618 in 2014
#     <br>
#     (b) Least number of casualties: 255 in 1971
#     <br>
#   (iii) Killed
#   <br>
#     (a) Most number of people killed: 44490 in 2014
#     <br>
#     (b) Least number of people killed: 173 in 1971
#     <br>
#   (iv) Wounded
#   <br>
#     (a) Most number of people wounded: 44043 in 2015
#     <br>
#     (b) Least number of people wounded: 82 in 1971

# 2. Region wise Attacks :
# <br>
#   (i) Attacks
#   <br>
#     (a) Most number of attacks: 50474 in "Middle East & North Africa"
#     <br>
#     (b) Least number of attacks: 282 in "Australasia & Oceania"
#     <br>
#   (ii) Casualties
#   <br>
#     (a) Most number of casualties: 351950 in "Middle East & North Africa"
#     <br>
#     (b) Least number of casualties: 410 in Australasia & Oceania
#     <br>
#   (iii) Killed
#   <br>
#     (a) Most number of people killed: 137642 in "Middle East & North Africa"
#     <br>
#     (b) Least number of people killed: 150 in "Australasia & Oceania"
#     <br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded: 214308 in "Middle East & North Africa"<br>
#     (b) Least number of people wounded: 260 in "Australasia & Oceania"<br>
# 

# 3. Country wise Attacks [Top 10] :
#     <br>
#   (i) Attacks<br>
#     (a) Most number of attacks: 24636 in "Iraq"<br>
#     (b) Least number of attacks: 4292 in "Turkey"<br>
#   (ii) Casualties<br>
#     (a) Most number of casualties: 213279 in "Iraq"<br>
#     (b) Least number of casualties: 22926 in "Philippines"<br>
#   (iii) Killed<br>
#     (a) Most number of people killed: 78589 in "Iraq"<br>
#     (b) Least number of people killed: 12053 in "El Salvador"<br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded: 134690 in "Iraq"<br>
#     (b) Least number of people wounded: 10328 in "Colombia"<br>

# 4. City wise Attacks [Top 10] :<br>
#   (i) Attacks<br>
#     (a) Most number of attacks: 7589 in "Baghdad"<br>
#     (b) Least number of attacks: 1019 in "Athens"<br>
#   (ii) Casualties<br>
#     (a) Most number of casualties: 77876 in "Baghdad"<br>
#     (b) Least number of casualties: 5748 in "Aleppo"<br>
#   (iii) Killed<br>
#     (a) Most number of people killed: 21151 in "Baghdad"<br>
#     (b) Least number of people killed: 2125 in "Aleppo"<br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded: 56725 in "Baghdad"<br>
#     (b) Least number of people wounded: 4955 in "Mogadishu"<br>

# 5. Terrorist Group wise Attacks [Top 10] :<br>
#   (i) Attacks<br>
#     (a) Most number of attacks : 7478 by "Taliban"<br>
#     (b) Least number of attacks : 2418 by "Boko Haram"<br>
#   (ii) Casualties<br>
#     (a) Most number of casualties : 69595 by "Islamic State of Iraq and the Levant (ISIL)"<br>
#     (b) Least number of casualties : 12130 by "Farabundo Marti National Liberation Front (FMLN)"<br>
#   (iii) Killed<br>
#     (a) Most number of people killed : 38923 by "Islamic State of Iraq and the Levant (ISIL)"<br>
#     (b) Least number of people killed : 5661 by "Revolutionary Armed Forces of Colombia (FARC)"<br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded : 30672 by "Islamic State of Iraq and the Levant (ISIL)"<br>
#     (b) Least number of people wounded : 4908 by "Kurdistan Workers' Party (PKK)"<br>

# 6. Attack Type wise Attacks:<br>
#   (i) Attacks<br>
#     (a) Most number of attacks : 88255 by "Bombing/Explosion"<br>
#     (b) Least number of attacks : 659 by "Hijacking"<br>
#   (ii) Casualties<br>
#     (a) Most number of casualties : 530007 by "Bombing/Explosion"<br>
#     (b) Least number of casualties : 7407 by "Facility/Infrastructure Attack"<br>
#   (iii) Killed<br>
#     (a) Most number of people killed : 160297 by "Armed Assault"<br>
#     (b) Least number of people killed : 880 by "Unarmed Assault"<br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded : 372686 by "Bombing/Explosion"<br>
#     (b) Least number of people wounded : 3765 by "Facility/Infrastructure Attack"<br>

# 7. Target Type wise Attacks:<br>
#   (i) Attacks<br>
#     (a) Most number of attacks : 43511 over "Private Citizens & Property"<br>
#     (b) Least number of attacks : 263 over "Abortion Related"<br>
#   (ii) Casualties<br>
#     (a) Most number of casualties : 319176 over "Private Citizens & Property"<br>
#     (b) Least number of casualties : 56 over "Abortion Related"<br>
#   (iii) Killed<br>
#     (a) Most number of people killed : 140504 over "Private Citizens & Property"<br>
#     (b) Least number of people killed : 10 over "Abortion Related"<br>
#   (iv) Wounded<br>
#     (a) Most number of people wounded : 178672 over "Private Citizens & Property"<br>
#     (b) Least number of people wounded : 46 over "Abortion Related"<br>
# 

# In[ ]:





# 비타민 6기 코테 - 기출

```python
import pandas as pd
import numpy as np
# 문제 1
a = []
b = []
for x in range(10):
  if x%2==0:
    a.append(x)
  else:
    b.append(x)
print(a, b)

# 문제 2 -> 자료형과 리스트, 튜플, 딕셔너리 정의 공부
a = 10
b = (1.44, 'bitamin')
c = "비타민"
d = [1, 2, 3, 4, 5]
e = {'특별시':'천안', '충남':'서울', '인천':'광역시'}
print(type(a))
print(type(b))
print(type(c))
print(type(d))
print(type(e))

# 리스트: 반복문을 이용해 쉽게 데이터를 관리할 수 있고 다양한 메서드를 제공한다. 
# 튜플: 리스트와 달리 ()를 사용하며 속도가 빠르고 한번 넣은 원소를 수정할 수 없습니다.  
# 딕셔너리: '키-값'이라는 쌍으로 데이터를 저장하므로 키를 이용하여 빠르게 해당 값을 찾을 수 있습니다.

# 문제 3
score = [90, 25, 67, 45, 80]
for x in score:
  if x>=80:
    print("우수")
  elif x>=60:
    print("보통")
  else:
    print("미흡")

import pandas as pd
import numpy as np
# 문제 4
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head(3)

df = titanic.loc[:,['pclass', 'fare']]
print(df)

# def z_score(x):
#   x = (x-min(x))/(max(x)-min(x))
#   return x

z_score = lambda x: (x-min(x))/(max(x)-min(x))

df.fare = z_score(df.fare)
df.head(3)

# 문제 5
# df = pd.read_csv('insurance2.csv')
# df.isnull()
# df = df.fillna(method='ffill')
# df = df.fillna(method='bfill')
```

# 비타민 7기 코테 - 재검토

```python
# 문제 1
a = [34, 30, 26, 40, 33, 15, 31, 21, 17, 40]
b = [45, 48, 25, 50, 50, 28, 39, 33, 47, 42]
c = [5, 10, 8, 10, 10, 7, 7, 9, 10, 2]
result = [sum(x) for x in zip(a, b, c)]
result.sort(reverse=True)
print(result)

# 문제 2
sn = input('지원번호를 입력하시오: ')

if sn[3] == '1' or sn[3] == '3':  
  print("성별: 남")
else:
  print("성별: 여")

tmp1 = (int(sn[:2])-17)*2-1
tmp2 = int(sn[3])
if tmp2 == 1 or tmp2 == 2:
  print("기수: %d" %(tmp1))
  print("편입: x")
else:
  print("기수: %d" %(tmp1-1))
  print("편입: o")

if len(sn)==7:
  print("운영진: o")
else:
  print("운영진: x")

# 문제 3
import pandas as pd # 패키지는 모두 설치되어 있다고 가정
import numpy as np

titanic = pd.read_csv('titanic.csv') # 3-1

df = titanic.loc[:100, ['Pclass', 'Age', 'Fare']] # 3-2

df = df.drop(['Pclass'], axis=1) # 3-3
display(df)

display(df.isnull().sum()) # 3-4

df = df.fillna(method='bfill') # 3-5

df = df[df['Age']>=40] # 3-6
df = df.sort_values(by=['Age'], ascending=False)
display(df)

magic = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(- x ** 2 / 2 ) # 3-7
df.Fare = magic(df.Fare)
display(df)

name_split = titanic["Name"].str.split(" ") # 3-8
df["First_Name"] = name_split.str.get(0)
display(df)
```
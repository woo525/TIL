### 리스트, 사전 연계
fruit = ["사과","사과", "바나나", "딸기","키위","복숭아","복숭아","복숭아"]
d = {}
for f in fruit:
    if f in d:
        d[f] = d[f] + 1
    else:
        d[f] = 1
print(d)


## 클래스, 오브젝트
# 클래스: 함수 + 변수 모아놓은 거
# 클래스: 빵틀, 오브젝트: 빵
class Person:
    def __init__(self, name, age): # self를 첫 인자로 받고
        self.name = name
        self.age = age
        
    def say_hello(self, to_name):
        print("안녕! " + to_name + " 나는" + self.name)
    
    def introduce(self):
        print("내 이름은 " + self.name + " 그리고 나는" + str(self.age) + "살이야")

woo = Person("우혁", 24)
woo.introduce()

class Police(Person):
    def arrest(self,to_arrest):
        print("넌 체포됐다, " + to_arrest)

class programmer(Person):
    def program(selfm to_program):
        print("다음엔 뭘 만들지? 아 이걸 만들어야겠다: " + to_program)

woo = Person("워니", 20)
jenny = Police("제니", 21)
michael = Programmer("마이클", 22)

jenny.introduce()
jenny.arrest("워니")
michael.introduce()
michael.program("이메일 클라이언트")


## 패키지와 모듈
# 패키지: 어떤 기능들을 구현하는 모듈들의 합 (라이브러리 == 패키지) (모듈 == 코드 들어있는 파일)
animal package
dog, cat modules
dog, cat modules cas say "hi"

from animal import dog # animal 패키지에서 dog 라는 모듈을 갖고와줘
from animal import cat # animal 패키지에서 cat 라는 모듈을 갖고와줘

from animal import * # animal 패키지가 갖고 있는 모듈을 다 불러와

d = dog.Dog() # instance
c = cat.Cat()

d.hi()
c.hi()


## geopy 가지고 패키지 다운 받아 사용하기
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="woohyeok")
location = geolocator.geocode("Seoul, South Korea")
print(location.raw)

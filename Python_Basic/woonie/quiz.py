# 퀴즈#3

 url = "http://naver.com"

 index = url.index("/")
 index1 = url.index("/",index + 1)
 index2 = url.index(".")

 # string = url.replace("http://","") -> 이렇게 해도 무방
 string = url[index1+1:index2]

 print(string[:3] + str(len(string)) + str(string.count("e")) + "!")

# 퀴즈#4
 from random import *
 ids= range(1, 21) # 1부터 20까지 숫자를 생성
 ids = list(ids)
 winners = sample(ids, 4)

 print(" -- 당첨자 발표 -- ")
 print("치킨 당첨자 : {0}".format(winners[0]))
 print("커피 당첨자 : {0}".format(winners[1:]))
 print(" -- 축하합니다 -- ")

# 퀴즈#5
 from random import *
 user = []
 cnt = 0
 for i in range(50):
     user.append(randrange(5,51))
     if 5 <= user[i] <= 15:
         cnt = cnt +1 
         print("[o] {0}번째 손님 (소요시간 : {1}분)".format(i+1,user[i]))
     else:
         print("[ ] {0}번째 손님 (소요시간 : {1}분)".format(i+1,user[i]))
 print("총 탑승 승객 : {0} 분".format(cnt))

# 퀴즈#6
 def std_weight(height, gender):
     if gender == "남자":
         key = 22
     else:
         key == 21
     result = height * height * key
     return result
 height = 175
 gender = "남자"
 weight = round(std_weight(height / 100, gender), 2) # 소수점 둘째자리에서 반올림
 print("키 {0}cm {1}의 표준 체중은 {2}kg 입니다.".format(height, gender, weight))

# 퀴즈#7
 import pickle
 for i in range (1,51):
     with open(str(i) + "주차.txt", "w", encoding="urf8") as report_file:
         report_file.write("- {0} 주차 주간보고 -".format(i))
         report_file.write("\n부서 : ")
         report_file.write("\n이름 : ")
         report_file.write("\n업무 요약 : ")
        
# 퀴즈#8
 class House:
     # 매물 초기화
     def __init__(self, location, house_type, deal_type, price, completion_year):
         self.location = location
         self.house_type = house_type
         self.deal_type = deal_type
         self.price = price
         self.completion_year = completion_year

     # 매물 정보 표시
     def show_detail(self):
         print(self.location, self.house_type, self.deal_type\
             ,self.price, self.completion_year)
        
 houses = []
 house1 = House("강남", "아파트", "매매", "10억", "2010년")
 house2 = House("마포", "오피스텔", "전세", "5억", "2007년")
 house3 = House("송파", "빌라", "월세", "500/50", "2000년")

 houses.append(house1)
 houses.append(house2)
 houses.append(house3)

 print("총 {0}대의 매물이 있습니다.".format(len(houses)))
 for house in houses:
     house.show_detail()









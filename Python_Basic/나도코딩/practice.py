# 문자열
 sentence = '나는 소년입니다.'
 print(sentence)
 sentence2 = "파이썬은 쉬워요."
 print(sentence2)
 sentence3 = """
 나는 소년이고,
 파이썬은 쉬워요
 """
 print(sentence3)

# 슬라이싱
 woo = "970525-1149413"
 print("성별: " + woo[7])
 print("연: " + woo[0:2]) # 0 부터 2 직전까지 (0 ~ 1)
 print("월: " + woo[2:4])
 print("일: " + woo[4:6])

 print("생년월일: " + woo[:6]) # 처음부터 6 직전까지
 print("뒤 7자리: " + woo[7:]) # 7번째 부터 끝까지
 print("뒤 7자리 (뒤에서부터): " + woo[-7:]) # 맨 뒤에서 7번째부터 끝까지

# 문자열 처리함수
 python = "Python is Amazing"
 print(python.lower())
 print(python.upper())
 print(python[0].isupper())
 print(len(python))
 print(python.replace("Python", "Java"))

 index = python.index("n")
 print(index)
 index = python.index("n",index + 1)
 print(index)

 print(python.find("Java")) # 해당 문자가 없으면 -1 반환
 # print(python.index("Java")) # 해당 문자가 없으면 오류 발생

 print(python.count("n"))

# 문자열 포맷
 방법 2
 print("나는 {}살입니다.".format(20))
 print("나는 {}색과 {}색을 좋아해요.".format("파란", "빨간"))
 print("나는 {0}색과 {1}색을 좋아해요.".format("파란", "빨간"))
 print("나는 {1}색과 {0}색을 좋아해요.".format("파란", "빨간"))

 방법 3
 print("나는 {age}살이며, {color}색을 좋아해요.".format(age=20, color="빨간"))
 print("나는 {age}살이며, {color}색을 좋아해요.".format(color="빨간", age=20))

 방법 4
 age = 20
 color = "빨간"
 print(f"나는 {age}살이며, {color}색을 좋아해요.")

# 탈출문자
 \n : 줄바꿈
 print("백문이 불여일견\n백견이 불여일타")

 # 저는 "나도코딩"입니다.
 print("저는 '나도코딩'입니다.")
 print('저는 "나도코딩"입니다.')
 print("저는 \"나도코딩\"입니다.")

 \\ : 문장 내에서 \
 print("C:\\Users\\lg\\Desktop\\PythonWorkspace>")

 \r : 커서를 맨 앞으로 이동
 print("Red Apple\rPine")

 \b : 백스페이스 (한 글자 = 삭제)
 print("Redd\bApple")

 \t : 탭
 print("Red\tApple")

# 리스트
 subway = ["유재석","조세호","박명수"]
 subway.append("하하")
 print(subway)

 정형돈씨를 유재석/ 조세호 사이에 태워봄
 subway.insert(1,"정형돈")
 print(subway)

 지하철에 있는 사람을 한명씩 뒤에서 꺼냄
 print(subway.pop())
 print(subway)

 특정 원소 개수 .count() / 정렬 .sort() / 뒤집기 .reverse() / 모두 지우기 .clear
 다양한 자료형 함께 사용 가능

 리스트 확장
 num_list = [5,4,3,2,1]
 mixlist = ["조세호", 20, True]
 num_list.extend(mixlist)
 print(num_list)

#사전
 cabinet = {3:"유재석", 100:"조세호"}
 print(cabinet[3]) # 찾는 키가 없으면 오류
 print(cabinet.get(3)) # 찾는 키가 없으면 "None"
 print(cabinet.get(3, "사용 불가"))
 print(3 in cabinet) # True

 추가, 삭제, 출력
 cabinet[3]="김종국"
 del cabinet[100]
 print(cabinet.keys())
 print(cabinet.values())
 print(cabinet.items())
 cabinet.clear()
 print(cabinet)

# 튜플 -> 변경되지 않는 목록을 활용할 때(속도가 리스트보다 빠름)
 menu = ("돈까스", "치즈까스")
 print(menu[0])
 print(menu[1])

 menu.add("생선까스") -> 튜플은 add 기능 제공하지 않음

 name = "김종국"
 age = 20
 hobby = "코딩"
 print(name, age, hobby)

 (name, age, hobby) = ("김종국", 20, "코딩")
 print(name, age, hobby)

 집합 (set)
 중복 안됨, 순서 없음
 my_set = {1,2,3,3,3}
 print(my_set)

 java = {"유재석", "김태호", "양세형"}
 python = set(["유재석", "박명수"]) # 리스트로 먼저 정의하고 집합으로 바꿔준 것

 교집합 (java, python 을 모두 할 수 있는 사람)
 print(java & python)
 print(java.intersection(python))

 합집합 (java 할 수 있거나 python 할 수 있는 개발자)
 print(java | python)
 print(java.union(python))

 차집합 (java 할 수 있지만 python 은 할 줄 모르는 사람)
 print(java - python)
 print(java.difference(python))

 python 할 줄 아는 사람이 늘어남
 python.add("김태호")
 print(python)

 java 를 잊었어요
 java.remove("김태호")
 print(java)

# 자료구조의 변경
 menu = {"커피", "우유", "주스"}
 print(menu, type(menu))

 menu = list(menu)
 print(menu, type(menu))

 menu = tuple(menu)
 print(menu, type(menu))

 menu = set(menu)
 print(menu,  type(menu))

 for문
 for waiting_no in [1,2,3,4,5]: # in range(1,6)
     print("대기번호 : {0}".format(waiting_no))

 한줄 for문 
 출석번호가 1,2,3,4 앞에 100을 붙이기로 함 -> 101, 102, 103, 104
 students = [1,2,3,4,5]
 print(students)
 students = [i+100 for i in students]
 print(students)

 학생 이름을 길이로 반환
 students = ["Iron man", "Thor", "I am groot"]
 students = [len(i) for i in students]
 print(students)

 학생 이름을 대문자로 변환
 students = ["Iron man", "Thor", "I am groot"]
 students = [i.upper() for i in students]
 print(students) 

# 함수
 def withdraw_night(balance, money): # 저녁에 출금
     commission = 100 # 수수료 100원
     return commission, balance - money - commission # 튜플 형태로 리턴

 balance = 0 # 잔액
 balance = 1000
 commission, balance = withdraw_night(balance, 500)
 print("수수료 {0} 원이며, 잔액은 {1} 원입니다.".format(commission, balance))

 기본값
 def profile(name, age, main_lang):
     print("이름 : {0}\t나이 : {1}\t주 사용 언어: {2}"\
         .format(name, age, main_lang))

 profile("유재석", 20, "파이썬")
 profile("김태호", 25, "자바")

 def profile(name, age=17, main_lang="파이썬"):
     print("이름 : {0}\t나이 : {1}\t주 사용 언어: {2}"\ # 코드가 길 경우
         .format(name, age, main_lang))

 profile("유재석")
 profile("김태호")

 키워드값
 def profile(name, age, main_lang):
     print(name, age, main_lang)

 profile(name="유재석", main_lang="파이썬", age=20)
 profile(main_lang="자바", age=25, name="김태호")

 가변 인자
 def profile(name, age, lang1, lang2, lang3, lang4, lang5):
     print("이름 : {0}\t나이 : {1}\t".format(name, age), end=" ") # 이렇게 끝내면 개행하지 않고 end 정의부 실행 후 종료
     print(lang1, lang2, lang3, lang4, lang5)

 def profile(name, age, *language):
     print("이름 : {0}\t나이 : {1}\t".format(name, age), end=" ") # 이렇게 끝내면 개행하지 않고 end 정의부 실행 후 종료
     for lang in language:
         print(lang, end=" ")
     print()

 profile("유재석", 20, "Python", "Java", "C", "C++", "C#", "JavaScript")
 profile("김태호", 25, "Kotlin", "Swift")

 지역변수와 전역변수
 gun = 10
 def checkpoint(soldiers): # 경계근무
     global gun # 전역 공간에 있는 gun 사용
     gun = gun - soldiers
     print("[함수 내] 남은 총 : {0}".format(gun))

 print("전체 총 : {0}".format(gun))
 checkpoint(2) # 2명이 경계 근무 나감
 print("남은 총 : {0}".format(gun))

# 표준 입출력

 print("Python", "Java", "JavaScript", sep =" vs ") # , 가 들어가는 부분에 문자 삽입
 print("Python", "Java", sep =",", end="?") # 이렇게 문장의 끝분을 정의해줄 수도 있음
 print(" 무엇이 제일 재미있을까요?")

 import sys
 print("Python", "Java", file=sys.stdout) # 로고처리 할 때 표준 출력은 문제 없음
 print("Python", "Java", file=sys.stderr) # 로고처리 할 때 에러로 인식

 시험 성적
 scores = {"수학":0, "영어":50, "코딩":100}
 for subject, score in scores.items():
     # print(subject, score)
     print(subject.ljust(8), str(score).rjust(4), sep=":") # n칸을 확보하고 왼쪽, 오른쪽 정렬

 은행 대기순번표
 001, 002, 003, ...
 for num in range(1, 21):
     print("대기 번호 : " + str(num).zfill(3))

 answer = input("아무 값이나 입력하세요 : ")
 print(type(answer)) # 사용자 입력을 통해 받은 데이터의 자료형은 str

 다양한 출력 포멧
 빈 자리는 빈공간으로 두고, 오른쪽 정렬을 하되, 총 10자리 공간을 확보
 print("{0: >10}".format(500))

 양수일 땐 +로 표시, 음수일 땐 -로 표시
 print("{0: >+10}".format(500))
 print("{0: >+10}".format(-500))

 왼쪽 정렬하고, 빈칸으로 _로 채움
 print("{0:_<+10}".format(500))

 3자리 마다 콤마를 찍어주기
 print("{0:,}".format(1000000000000000))

 3자리 마다 콤마를 찍어주기, +- 부호도 붙이기
 print("{0:+,}".format(1000000000000000))

 3자리 마다 콤마를 찍어주기, 부호도 붙이고, 자릿수 확보하기
 돈이 많으면 행복하니까 빈 자리는 ^ 로 채워주기
 print("{0:^<+30,}".format(1000000000000000))

 소수점 출력
 print("{0:f}".format(5/3))

 소수점 특정 자리수 까지만 표시 (소수점 3째 자리에서 반올림)
 print("{0:.2f}".format(5/3))

 파일 입출력
 score_file = open("score.txt", "w", encoding="utf8")
 print("수학 : 0", file=score_file)
 print("영어 : 50", file=score_file)
 score_file.close()

 score_file = open("score.txt", "a", encoding="utf8")
 score_file.write("과학 : 80")
 score_file.write("\n코딩 : 100")
 score_file.close()

 score_file = open("score.txt", "r", encoding="utf8")
 print(score_file.read())
 score_file.close()

 score_file = open("score.txt", "r", encoding="utf8")
 print(score_file.readline(), end="") # 줄별로 읽기, 한 줄 읽고 커서는 다음 줄로 이동
 print(score_file.readline(), end="")
 print(score_file.readline(), end="")
 print(score_file.readline(), end="")
 score_file.close()

 score_file = open("score.txt", "r", encoding="utf8")
 while True:
     line = score_file.readline()
     if not line:
        break
     print(line, end="")
 score_file.close() 

 score_file = open("score.txt", "r", encoding="utf8")
 lines = score_file.readlines() # list 형태로 저장
 for line in lines:
     print(line, end="")
 score_file.close()

 pickle
 import pickle
 profile_file = open("profile.pickle", "wb")
 profile = {"이름":"박명수", "나이":30, "취미":["축구", "골프", "코딩"]}
 print(profile)
 pickle.dump(profile, profile_file) # profile 에 있는 정보를 file 에 저장
 profile_file.close()

 profile_file = open("profile.pickle", "rb")
 profile = pickle.load(profile_file) # file 에 있는 정보를 profile 에 불러오기
 print(profile)
 profile_file.close()

 with open("profile.pickle", "rb") as profile_file:
     print(pickle.load(profile_file))

 with open("study.txt", "w", encoding="urf8") as study_file:
     study_file.write("파이썬을 열심히 공부하고 있어요")

 with open("study.txt", "r", encoding="urf8") as study_file:
     print(study_file.read())

# 클래스 = 붕어빵틀 -> 함수와 변수의 집합
 ## 클래스를 사용하지 않을 경우
 # 마린 유닛
 name = "마린"
 hp = 40
 damage = 5

 print("{} 유닛이 생성되었습니다.".format(name))
 print("체력 {0}, 공격력 {1}\n".format(hp, damage))

 # 탱크 유닛
 tank_name = "탱크"
 tank_hp = 150
 tank_damage = 35

 print("{} 유닛이 생성되었습니다.".format(tank_name))
 print("체력 {0}, 공격력 {1}\n".format(tank_hp, tank_damage))

 # 탱크 유닛2
 tank_name2 = "탱크"
 tank_hp2 = 150
 tank_damage2 = 35

 print("{} 유닛이 생성되었습니다.".format(tank_name2))
 print("체력 {0}, 공격력 {1}\n".format(tank_hp2, tank_damage2))

 def attack(name, location, damage):
     print("{0} : {1} 방향으로 적군을 공격 합니다. [공격력 {2}]".format(\
         name, location, damage))

 attack(name, "1시", damage)
 attack(tank_name, "1시", tank_damage)
 attack(tank_name, "1시", tank_damage2)

 클래스를 사용할 경우
 일반 유닛
 class Unit:
     def __init__(self, name, hp, damage): # __init__ = 생성자
         self.name = name # name = 멤버 변수 
         self.hp = hp
         # self.damage = damage
         # print("{} 유닛이 생성되었습니다.".format(self.name))
         # print("체력 {0}, 공격력 {1}\n".format(self.hp, self.damage))

 marine1 = Unit("마린", 40, 5) # marine = 객체
 marine2 = Unit("마린", 40, 5)
 tank = Unit("탱크", 150, 35) 

 wraith1 = Unit("레이스", 80, 5)
 wraith2 = Unit("빼앗은 레이스", 80, 5)
 wraith2.clocking = True # 파이썬에서는 객체에 외부에서 추가로 변수 생성 가능

 if wraith2.clocking == True:
     print("{0} 는 현재 클로킹 상태입니다.".format(wraith2.name))

 공격 유닛
 class AttackUnit:
     def __init__(self, name, hp, damage): # __init__ = 생성자
         self.name = name # name = 멤버 변수 
         self.hp = hp
         self.damage = damage

     def attack(self, location):
         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"\
             .format(self.name, location, self.damage)) # 클래스 안에서 메소드 첫 인자는 무조건 self 라고 생각

     def damaged(self, damage):
         print("{0} : {1} 데미지를 입었습니다. ".format(self.name, damage))
         self.hp -= damage
         print("{0} : 현재 체력은 {1} 입니다. ".format(self.name, self.hp))
         if(self.hp<=0):
             print("{0} : 파괴되었습니다.".format(self.name))

 firebat1 = AttackUnit("파이어뱃", 50, 16)
 firebat1.attack("5시")

 # 공격 2번 받는다고 가정
 firebat1.damaged(25)
 firebat1.damaged(25)

# 상속 : 클래스의 내용이 중복될 경우 사용

 class Unit:
     def __init__(self, name, hp): # __init__ = 생성자
         self.name = name # name = 멤버 변수 
         self.hp = hp

 class AttackUnit(Unit): # Unit 함수를 상속 받는다
     def __init__(self, name, hp, damage): # __init__ = 생성자
         Unit.__init__(self, name, hp)
         self.damage = damage

     def attack(self, location):
         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"\
             .format(self.name, location, self.damage)) # 클래스 안에서 메소드 첫 인자는 무조건 self 라고 생각

     def damaged(self, damage):
         print("{0} : {1} 데미지를 입었습니다. ".format(self.name, damage))
         self.hp -= damage
         print("{0} : 현재 체력은 {1} 입니다. ".format(self.name, self.hp))
         if(self.hp<=0):
             print("{0} : 파괴되었습니다.".format(self.name))

 firebat1 = AttackUnit("파이어뱃", 50, 16)
 firebat1.attack("5시")

 # 공격 2번 받는다고 가정
 firebat1.damaged(25)
 firebat1.damaged(25)

# 다중 상속

 class Unit:
     def __init__(self, name, hp): # __init__ = 생성자
         self.name = name # name = 멤버 변수 
         self.hp = hp

 class AttackUnit(Unit): # Unit 함수를 상속 받는다
     def __init__(self, name, hp, damage): # __init__ = 생성자
         Unit.__init__(self, name, hp)
         self.damage = damage

     def attack(self, location):
         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"\
             .format(self.name, location, self.damage)) # 클래스 안에서 메소드 첫 인자는 무조건 self 라고 생각

     def damaged(self, damage):
         print("{0} : {1} 데미지를 입었습니다. ".format(self.name, damage))
         self.hp -= damage
         print("{0} : 현재 체력은 {1} 입니다. ".format(self.name, self.hp))
         if(self.hp<=0):
             print("{0} : 파괴되었습니다.".format(self.name))

 class Flyable:
     def __init__(self, flying_speed):
         self.flying_speed = flying_speed
    
     def fly(self, name, location):
         print("{0} : {1} 방향으로 날아갑니다. [속도 {2}]"\
             .format(name, location, self.flying_speed))

 class FlyableAttackUnit(AttackUnit, Flyable):
     def __init__(self, name, hp, damage, flying_speed):
         AttackUnit.__init__(self, name, hp, damage)
         Flyable.__init__(self, flying_speed)

 valkyrie = FlyableAttackUnit("발키리", 200, 6, 5)
 valkyrie.fly(valkyrie.name, "3시")

# 오버라이딩 : 자식클래스에서 정의한 메소드를 사용하고 싶을 때

 class Unit:
     def __init__(self, name, hp, speed): 
         self.name = name 
         self.hp = hp
         self.speed = speed
    
     def move(self, location):
         print("[지상 유닛 이동]")
         print("{0} : {1} 방향으로 이동합니다. [속도 {2}]"\
             .format(self.name, location, self.speed))

 class AttackUnit(Unit):
     def __init__(self, name, hp, speed, damage):
         Unit.__init__(self, name, hp, speed)
         self.damage = damage

     def attack(self, location):
         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"\
             .format(self.name, location, self.damage))

     def damaged(self, damage):
         print("{0} : {1} 데미지를 입었습니다. ".format(self.name, damage))
         self.hp -= damage
         print("{0} : 현재 체력은 {1} 입니다. ".format(self.name, self.hp))
         if(self.hp<=0):
             print("{0} : 파괴되었습니다.".format(self.name))

 class Flyable:
     def __init__(self, flying_speed):
         self.flying_speed = flying_speed
    
     def fly(self, name, location):
         print("{0} : {1} 방향으로 날아갑니다. [속도 {2}]"\
             .format(name, location, self.flying_speed))

 class FlyableAttackUnit(AttackUnit, Flyable):
     def __init__(self, name, hp, damage, flying_speed):
         AttackUnit.__init__(self, name, hp, 0, damage) # 지상 스피드는 0
         Flyable.__init__(self, flying_speed)

     def move(self, location):
         print("[공중 유닛 이동]")
         self.fly(self.name, location)

 vulture = AttackUnit("벌쳐", 80, 10, 20)
 battlecruiser = FlyableAttackUnit("배틀크루저", 500, 25, 3)

 # 이럴 경우 매번 move 와 fly 를 구분해가며 사용해야하는 불편
 vulture.move("11시")
 # battlecruiser.fly(battlecruiser.name, "9시")

 battlecruiser.move("9시")

# pass : 아무것도 안하고 일단 넘어감
 class BuildingUnit(Unit):
     def __init__(self, name, hp, location):
         pass

 suppply_depot = BuildingUnit("서플라이 디폿", 500, "7시")

 def game_start():
     print("[알림] 새로운 게임을 시작합니다.")

 def game_over():
     pass

 game_start()
 game_over()

# super
 class BuildingUnit(Unit):
     def __init__(self, name, hp, location):
         #Unit.__init__(self, name, hp, 0)
         super().__init__(name, hp, 0) # 윗줄 대신 super 사용 가능. () 넣고 self 빼고
         self.location = location

# 스타크래프트 프로젝트 전반전

 from random import *

 class Unit:
     def __init__(self, name, hp, speed): 
         self.name = name 
         self.hp = hp
         self.speed = speed
         print("{0} 유닛이 생성되었습니다.".format(name))
    
     def move(self, location):
         print("{0} : {1} 방향으로 이동합니다. [속도 {2}]"\
             .format(self.name, location, self.speed))

     def damaged(self, damage):
         print("{0} : {1} 데미지를 입었습니다. ".format(self.name, damage))
         self.hp -= damage
         print("{0} : 현재 체력은 {1} 입니다. ".format(self.name, self.hp))
         if(self.hp<=0):
             print("{0} : 파괴되었습니다.".format(self.name))

 class AttackUnit(Unit):
     def __init__(self, name, hp, speed, damage):
         Unit.__init__(self, name, hp, speed)
         self.damage = damage

     def attack(self, location):
         print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"\
             .format(self.name, location, self.damage))

 # 마린
 class Marine(AttackUnit):
     def __init__(self):
         AttackUnit.__init__(self, "마린", 40, 1, 5) 

     # 스팀팩
     def stimpack(self):
         if self.hp > 10:
             self.hp -= 10
             print("{0} : 스팀팩을 사용합니다. (HP 10 감소)".format(self.name))
         else:
             print("{0} : 체력이 부족하여 스팀팩을 사용하지 않습니다.".format(self.name))

 # 탱크
 class Tank(AttackUnit):
     # 시즈모드
     seize_developed = False # 시즈모드 개발여부
    
     def __init__(self):
         AttackUnit.__init__(self, "탱크", 150, 1, 35)
         self.seize_mode = False

     def set_seize_mode(self):
         if Tank.seize_developed == False:
             return

         # 현재 시즈모드가 아닐 때
         if self.seize_mode == False:
             print("{0} : 시즈모드로 전환합니다.".format(self.name))
             self.damage *= 2
             self.seize_mode = True
         # 현재 시즈모드일 때
         else:
             print("{0} : 시즈모드를 해제합니다.".format(self.name))
             self.damage /= 2
             self.seize_mode = False

 class Flyable:
     def __init__(self, flying_speed):
         self.flying_speed = flying_speed
    
     def fly(self, name, location):
         print("{0} : {1} 방향으로 날아갑니다. [속도 {2}]"\
             .format(name, location, self.flying_speed))

 class FlyableAttackUnit(AttackUnit, Flyable):
     def __init__(self, name, hp, damage, flying_speed):
         AttackUnit.__init__(self, name, hp, 0, damage) # 지상 스피드는 0
         Flyable.__init__(self, flying_speed)

     def move(self, location): 
         self.fly(self.name, location)

 # 레이스
 class Wraith(FlyableAttackUnit):
     def __init__(self):
         FlyableAttackUnit.__init__(self, "레이스", 80, 20, 5)
         self.clocked = False # 클로킹 모드 (해제 상태)

     def clocking(self):
         if self.clocked == True: # 클로킹 모드 -> 해제
             print("{0} : 클로킹 모드 설정합니다.".format(self.name))
             self.clocked = True

 ### 스타크래프트 후반전
 def game_start():
     print("[알림] 새로운 게임을 시작합니다.")

 def game_over():
     print("Player : gg") # good game
     print("[Player] 님이 게임에서 퇴장하셨습니다.")

 # 실제 게임 진행
 game_start()

 # 마린 3기 생성
 m1 = Marine()
 m2 = Marine()
 m3 = Marine()

 # 탱크 2기 생성
 t1 = Tank()
 t2 = Tank()

 # 레이스 1기 생성
 w1 = Wraith()

 # 유닛 일괄 관리 (생성된 모든 유닛 append)
 attack_units = []
 attack_units.append(m1)
 attack_units.append(m2)
 attack_units.append(m3)
 attack_units.append(t1)
 attack_units.append(t2)
 attack_units.append(w1)

 # 전군 이동
 for unit in attack_units:
     unit.move("1시")

 # 탱크 시즈모드 개발
 Tank.seize_developed = True
 print("[알림] 탱크 시즈 모드 개발이 완료되었습니다.")

 # 공격 모드 준비 (마린 : 스팀팩, 탱크 : 시즈모드, 레이스 : 클로킹)
 for unit in attack_units:
     if isinstance(unit, Marine): # 지금 만들어진 객체가 특정 클래스의 인스턴스인지 확인 
         unit.stimpack()
     elif isinstance(unit, Tank):
         unit.set_seize_mode()
     elif isinstance(unit, Wraith):
         unit.clocking()

 # 전군 공격
 for unit in attack_units:
     unit.attack("1시")

 # 전군 피해
 for unit in attack_units:
     unit.damaged(randint(5,21)) # 공격은 랜덤으로 받음(5 ~ 20)

 # 게임 종료
 game_over()

# 예외처리

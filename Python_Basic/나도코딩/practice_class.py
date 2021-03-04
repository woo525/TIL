class Unit:
    def __init__(self):
        print("Unit 생성자")

class Flyable:
    def __init__(self):
        print("Flyable  생성자")

class FlyableUnit(Flyable, Unit):
    def __init__(self):
        #super().__init__() # 이렇게 하면 맨 처음 상속 받는 클래스에 대해서만 초기화 함수 호출
        Unit.__init__(self)
        Flyable.__init__(self)

dropship = FlyableUnit()
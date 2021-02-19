import random #加個注釋
class Pig(object):
    def __init__(self): #初始化
        self.number=list(range(1,50)) #產生1~49數字
        self.combo_number=list() #
        self.user_combo=list() #存用戶已經中獎號碼
        self.big_number=list() #存機器開獎號碼
        self.specially_number=0 #存特殊號碼

    def user_number(self,one,two,three,four,five,six): #用戶自己選擇6個數字
        self.combo_number.append(one)
        self.combo_number.append(two)
        self.combo_number.append(three)
        self.combo_number.append(four)
        self.combo_number.append(five)
        self.combo_number.append(six)
        print('你選擇的六個號碼為{0}'.format(self.combo_number))

    def user_random_number(self): #機器幫用戶隨機選擇6個數字
        self.combo_number=random.sample(self.number,6)
        print("機器幫你選擇的六個號碼為:{0}".format(self.combo_number))
        return self.combo_number

    def seven_number(self): # 大樂透中獎的7個號碼
        self.big_number = random.sample(self.number, 7)  # 大樂透中獎的7個號碼
        self.specially_number = random.sample(self.big_number, 1)  # 從已中獎的7個號碼抽一個當特別號
        for _ in self.specially_number:
            self.big_number.remove(_)  # 刪除原本7個號碼中的特別號
        print('大樂透本期開獎號碼為: {0},特別號碼為: {1}'.format(self.big_number, self.specially_number))
        return self.big_number

    def user_combo_number(self):#對號碼
        for i in self.combo_number:  # 循環用戶選的號碼
            if i in self.big_number:  # 如果用戶選的這個號碼是存在於中獎號碼中
                self.user_combo.append(i)  # 就添加到已中獎號碼中
        return self.user_combo

    def len_number(self):#開獎
        count_number = len(self.user_combo)  # 已中獎號碼長度
        combo_number_sort = sorted(self.user_combo)  # 排序後的中獎號碼
        if sorted(self.combo_number) == sorted(self.big_number):  # 如果排序後的號碼等於排序后的大樂透號碼就是頭獎
            print("恭喜你中了頭將")
        elif self.specially_number in self.user_combo: #如果特殊號也包含在已經中獎號碼中
            if count_number == 6:  # 對中當期獎號之任五碼＋特別號
                print('恭喜你中了貳獎')
                print("你的中將號碼為{0},中獎的特別號為{1}".format(combo_number_sort, self.specially_number))
            elif count_number == 5:  # 對中當期獎號之任四碼＋特別號
                print("恭喜你中了肆獎")
                print("你的中將號碼為{0},中獎的特別號為{1}".format(combo_number_sort, self.specially_number))
            elif count_number == 4:  # 對中當期獎號之任三碼＋特別號
                print("恭喜你中了陸獎 NT$1,000")
                print("你的中將號碼為{0},中獎的特別號為{1}".format(combo_number_sort, self.specially_number))
            elif count_number == 3:  # 對中當期獎號之任兩碼＋特別號
                print("恭喜你中了柒獎 NT$400")
                print("你的中將號碼為{0},中獎的特別號為{1}".format(combo_number_sort, self.specially_number))
        else: #已經中獎號碼中沒有特殊號碼
            if count_number == 0:
                print("你一個號碼都沒中,歡迎下次光臨")
            elif count_number == 5:
                print("恭喜你中了參獎")
                print("你的中將號碼為{0}".format(combo_number_sort))
            elif count_number == 4:
                print("恭喜你中了伍獎 NT$2,000")
                print("你的中將號碼為{0}".format(combo_number_sort))
            elif count_number == 3:
                print("恭喜你中了普獎 NT$400")
                print("你的中將號碼為{0}".format(combo_number_sort))
            elif count_number <= 2:
                print("恭喜你中了{0}個號碼,你的中獎號碼為{1},但是沒有中獎".format(count_number, combo_number_sort))

if __name__ == '__main__':
    for i in range(100):
        print("第{0}次開獎".format(i))
        a=Pig()
        a.user_random_number() #機器幫用戶隨機選擇號碼
        # a.user_number(48,5,20,10,32,21) #自己選號碼
        a.seven_number() #打印機器開獎的號碼
        a.user_combo_number() #打印用戶有中獎的號碼
        a.len_number() #對講號碼
        print("-"*50)



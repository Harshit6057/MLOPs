class Calculator:
    def __init__(self):
        self.num1 = int(input("Enter the First number: "))
        self.num2 = int(input("Enter the Second number: "))

    
    def add(self):
        r = self.num1 + self.num2
        result = f"The sum of {str(self.num1)} and {str(self.num2)} is {str({r})}"
        print(result)

    def subtract(self):
        r = self.num1 - self.num2
        result = f"The subtract of {str(self.num1)} and {str(self.num2)} is {str({r})}"
        print(result)

    
    def multiply(self):
        r = self.num1 * self.num2
        result = f"The multiply of {str(self.num1)} and {str(self.num2)} is {str({r})}"
        print(result)


    def divide(self):
        r = self.num1 / self.num2
        result = f"The division of {str(self.num1)} and {str(self.num2)} is {str({r})}"
        print(result)


if __name__ == '__main__':
    cal = Calculator()
    cal.add()
    cal.subtract()
    cal.multiply()
    cal.divide()

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Ansh'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="CGPA must be between 0 and 10")

# new_student = {'name' : 'Ansh'}
new_student = {'age' : 24, 'email' : 'ansh@gmail.com', 'cgpa' : 8.5}

student = Student(**new_student)
# "student" is a Pydantic Object

# Converting Pydantic Object to Dictionary
student_dict = dict(student)
print(student_dict['age'])

# Converting Pydantic Object to JSON
student_json = student.model_dump_json()
print(student_json)
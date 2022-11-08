from app import db

class Parameter_vector(db.Model):
    par = db.Column(db.Integer, primary_key=True)

class Person():
    def __init__(self, id, group, profile=None):
        self.id = id
        self.group = group
        if self.group=='g1':
            temp = 'child'
        else:
            temp = 'adult'
        self.name = 'Person {} ({})'.format(self.id+1, temp)
        self.profile = profile


    def add_profile(self, profile):
        self.profile = profile
        return None

class Bedroom():
    def __init__(self, id, name):
        self.id = id
        self.name = name

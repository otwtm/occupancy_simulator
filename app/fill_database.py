from app import db
# db.create_all()
from app import Parameter_vector
par1 = Parameter_vector(par=65)
db.session.add(par1)
db.session.commit()
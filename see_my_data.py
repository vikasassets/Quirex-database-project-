from querix.engine import QuerixEngine

db = QuerixEngine("my_permanent_data")
db.put("owner_name", "The vikas Kumar is the owner of this project")
db.close()
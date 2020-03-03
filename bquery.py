import pandas

query = 'SELECT * FROM [project-999:dataset.table]'
project_id = 'project-999'
private_key = '/path.to/private_key.json'  # path or JSON format, either one is fine!

pandas.read_gbq(query, project_id=project_id, private_key=private_key)
from neo4j import GraphDatabase

def clear_neo4j_db(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

    
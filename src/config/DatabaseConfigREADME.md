# Database Configuration Guide

This guide explains how to configure connection parameters for each database. Please complete the database installation described in [README](./docker/README.md).  first.

**Configuration File Location**: `src/config/config.ini`

## Configuration Structure

Each database configuration includes the following parameters:

| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| host | 127.0.0.1 | Database server IP (can remain 127.0.0.1 for local development) |
| front_end_port | 8080 | Port for frontend visualization tools |
| read_write_port | 3306 | Port for program read/write operations |
| username | root | Database login account |
| password | (corresponding password) | Database login password |

## Configuration Examples

```ini
[Elasticsearch]
host = 127.0.0.1
front_end_port = 5601
read_write_port = 9200
username = elastic
password = elastic

[Milvus]
host = 127.0.0.1
front_end_port = 8000
read_write_port = 19530
username = 
password = 
min_content_len = 200

[Neo4j]
host = 127.0.0.1
front_end_port = 7474   
read_write_port = 7687
username = neo4j
password = neo4j2025

[MySQL]
host = 127.0.0.1
front_end_port = 8080
read_write_port = 3306
username = root
password = 123456
```

### Database-Specific Notes

#### Elasticsearch

* Default user: `elastic`
* Default password: `elastic`
* Port usage:
  * 9200: Read/write port
  * 5601: Kibana visualization web frontend port

#### Milvus

* Special parameter:
  * `min_content_len = 200`: Sets the minimum length (in characters) for inserted text chunks
* Port usage:
  * 19530: Read/write port
  * 8000: Attu visualization web frontend port

#### Neo4j

* Default user: `neo4j`
* Default password: `neo4j2025`
* Port usage:
  * 7687: Read/write port using Bolt protocol
  * 7474: Neo4j visualization web frontend port

#### MySQL

* Default user: `root`
* Default password: `123456`
* Port usage:
  * 3306: Read/write port
  * 8080: Adminer visualization web frontend port

# Backend Port Configuration Guide

This guide explains the format and usage of backend port configuration.

**Configuration File Location**: `src/config/config.ini`

## Configuration Structure

Backend port parameters are configured under `[backend_api]`, setting the name of each service and its corresponding port number.

## Configuration Example

```ini
[backend_api]
data_search_port = 1242
deepwriter_port = 1244
deepsearch_port = 1246
```

When running `src/backend/data_search_services.py`, the service will start on port 1242 as specified by `data_search_port`.

> **Note**: For specific backend running examples, see :ref:`examples_backend`.

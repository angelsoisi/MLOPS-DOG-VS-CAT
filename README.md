# 1. Stop current containers
docker-compose down

# 2. Build the image with the dependencies 
docker-compose build --no-cache

# 3. Start again
docker-compose up -d

# 4. Access logs 
docker-compose logs -f

# verify is working
curl http://localhost/health

# see all containers 
docker-compose ps

# see the logs on live time
docker-compose logs -f

# See the logs from a specific node 
docker-compose logs -f api_node1
docker-compose logs -f nginx

# See which node answers the petitions 
curl -I http://localhost/health | grep X-Upstream-Server

# Do multiple requests and see the distribution
for i in {1..20}; do
  curl -s -I http://localhost/health | grep X-Upstream-Server
done

# Stop a node to verify the auto-recovery
docker-compose stop api_node1

# Nginx will automatically redirect to the other 5 nodes 

# Start the node again
docker-compose start api_node1

# Volver a iniciar el nodo
docker-compose start api_node1

# Detecting dogs 
<img width="614" height="325" alt="image" src="https://github.com/user-attachments/assets/cfbf16e9-6173-44a8-b2b9-8d8884b915ef" />

# Detecting cats 

<img width="610" height="326" alt="image" src="https://github.com/user-attachments/assets/0e2c464a-13cf-482d-9d8c-d8d673382a4a" />

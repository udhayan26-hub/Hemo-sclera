# Use official Node.js image as base
FROM node:18-alpine AS base

# Install python and build dependencies
RUN apk add --no-cache python3 make g++ py3-pip

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build the Next.js application
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]

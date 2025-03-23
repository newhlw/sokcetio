const express = require("express");
const { createServer } = require("node:http");

const { Server } = require("socket.io");
const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*", // Allow all domains (replace with specific domains if needed)
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"],
    credentials: true, // Set to true if you need to support cookies or credentials
  },
});

app.set("view engine", "ejs");
io.on("connection", (socket) => {
  socket.emit("init", socket.id);
  console.log(`a userc conencted`);
  socket.on("send-to", ({ from, id, msg }) => {
    io.to(id).emit("reverse", { from, id: id, msg: msg });
  });
  socket.on("massage", (msg) => {
    console.log(`massage sent from clint: ${msg}`);
    socket.broadcast.emit("reverse", msg);
  });
  socket.on("joinroom", (roomid) => {
    socket.join(roomid);
    socket.to.emit("new-user", { id: socket.id });
  });
  socket.on("join", (code) => {
    socket.join(code);
  });
  socket.on("chat", ({ room, msg }) => {
    socket.broadcast.to(room).emit("chat-backend", msg);
  });
 
});


server.listen(3000, () => {
  console.log("server running at http://localhost:3000");
});

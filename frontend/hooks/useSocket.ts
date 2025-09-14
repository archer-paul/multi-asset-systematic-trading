'use client'

import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

const useSocket = (uri) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketInstance = io(uri, {
      transports: ['websocket'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    setSocket(socketInstance);

    socketInstance.on('connect', () => {
      console.log('Socket connected!');
      setIsConnected(true);
    });

    socketInstance.on('disconnect', () => {
      console.log('Socket disconnected.');
      setIsConnected(false);
    });

    return () => {
      socketInstance.disconnect();
    };
  }, [uri]);

  return { socket, isConnected };
};

export default useSocket;

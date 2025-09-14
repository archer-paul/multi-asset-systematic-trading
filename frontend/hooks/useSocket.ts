'use client'

import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface UseSocketProps {
  uri?: string;
}

const useSocket = (uri?: string) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketUri = uri || process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:5000';

    const socketInstance = io(socketUri, {
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

    socketInstance.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      setIsConnected(false);
    });

    return () => {
      socketInstance.disconnect();
    };
  }, [uri]);

  return { socket, isConnected };
};

export { useSocket };
export default useSocket;

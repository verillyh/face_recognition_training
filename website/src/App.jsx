import { useEffect, useState } from 'react'
import { io } from "socket.io-client"
import './index.css'

const socket = io("http://localhost:5000")

function App() {
  const [imageSrc, setImageSrc] = useState("")
  const [personName, setPersonName] = useState("No faces detected")
  const [registerName, setRegisterName] = useState("")

  useEffect(() => {
    socket.on("frame", data => {
      setImageSrc("data:image/jpeg;base64," + data)
    })
    socket.on("detected", data => {
      setPersonName(data)
      
    })
  }, [])

  function registerFace() {
    if (registerName == "") {
      setRegisterName("Enter name here")
    }
    socket.emit("register", registerName)
  }
  function stopFeed() {
    socket.emit("stop")
  }

  return (
    <>
      <main className='min-h-full w-full flex flex-col items-center pt-10'>
        <img src={imageSrc} className='w-xl'/>
        <div className="flex flex-col items-center">
          <p className='mt-3'>{personName}</p>
          <input type="text" value={registerName} onChange={e => setRegisterName(e.target.value)} placeholder='Person name...' className='block w-full mt-10 rounded-md bg-white px-3 py-1 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-sm/6'/>
          <button type="button" onClick={registerFace} className="flex w-full mt-5 justify-center rounded-md bg-emerald-600 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-xs hover:bg-emerald-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600">Register face</button>
          <button type="button" onClick={stopFeed} className="flex w-full mt-5 justify-center rounded-md bg-rose-600 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-xs hover:bg-rose-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-600">Stop feed</button>
        </div>
      </main>
    </>
  )
}

export default App

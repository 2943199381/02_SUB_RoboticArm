# USB VCP C++ Driver (Linux)

这是一个主机侧 C++ 驱动，用于接收 STM32 VCP 数据并解析带 `cmd_id`、`CRC16` 的数据帧。

## 帧格式

- 字节 0：`SOF`，固定 `0xA5`
- 字节 1~2：`len`，2 字节小端，表示 `payload` 长度
- 字节 3~4：`cmd_id`，2 字节小端
- 字节 5~(5+len-1)：`payload`
- 最后 2 字节：`crc16`（`CRC16-Modbus`，校验范围 `frame[0..end-3]`）

总长度：`1 + 2 + 2 + len + 2 = 7 + len`

解析流程（简版）：查找 `SOF` → 读取 `len` → 判断是否收满 `7 + len` 字节 → 校验 `CRC16` → 输出 `cmd_id/payload`。

## 发送与接收线程

驱动仅保留直接发送与接收能力：

- 发送：`sendFrame(...)`
- 接收：`pollOnce(...)` 或独立线程 `startReceiveThread(callback)` / `stopReceiveThread()`

建议在高频场景下使用独立接收线程，避免主循环中发送逻辑影响接收吞吐。

## 文件

- `UsbVcpDriver.hpp`: 驱动头文件
- `UsbVcpDriver.cpp`: 驱动实现
- `example_main.cpp`: 示例程序

## 编译

```bash
cmake -S . -B build
cmake --build build -j
```

## 运行

```bash
./usb_recv /dev/ttyACM0 115200
```

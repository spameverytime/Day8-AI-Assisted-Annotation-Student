# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 18 xe

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
- Vị trí 1 (dễ bỏ sót hoặc cắt cụt): Góc dưới bên phải có một xe chạy sát làn trong bị mép ảnh cắt mất một phần thân xe, chỉ còn nhìn thấy phần đuôi và cụm đèn hậu. AI dễ bỏ sót hoặc chỉ nhận diện một phần không ôm trọn thân xe theo guideline.
- Vị trí 2 (dễ bỏ sót hoặc nhận diện sai): Khu vực phía xa gần chân cầu vượt có nhiều xe chạy nối đuôi nhau ở cự ly xa chỉ còn hai chấm đèn nhỏ, AI dễ bỏ sót do kích thước nhỏ hoặc nhận nhầm các vệt phản quang ánh đèn pha trên mặt đường bên trái thành ô tô.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.

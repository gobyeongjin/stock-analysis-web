def normalize_code(value):
    code = str(value).strip().removeprefix("A")

    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"잘못된 종목코드: {value}")

    return code


def validate_buy_orders(
    orders,
    held_codes,
    pending_buy_codes,
    max_positions=2,
):
    held = {normalize_code(x) for x in held_codes}
    pending = {normalize_code(x) for x in pending_buy_codes}

    # 주문 대기 중인 종목도 포지션 슬롯을 차지함
    occupied = held | pending

    if len(occupied) > max_positions:
        raise RuntimeError("보유 + 미체결 종목이 최대 슬롯 초과")

    planned = [normalize_code(x) for x in orders]

    if len(planned) != len(set(planned)):
        raise RuntimeError("중복 매수 계획")

    duplicates = set(planned) & occupied

    if duplicates:
        raise RuntimeError(
            f"보유 또는 미체결 종목 중복: {sorted(duplicates)}"
        )

    if len(occupied | set(planned)) > max_positions:
        raise RuntimeError("최대 2종목 제한 초과")

    return True

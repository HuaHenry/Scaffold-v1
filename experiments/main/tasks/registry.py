"""One historical anchor and sixteen mutually assigned granular/fluid scene templates."""
from ..shared import integer

TASKS = {
    "T01": {"name": "现有结构干预基准", "family": "structure", "template": "legacy03",
            "boundary": "完整沿用03的simple_to_complex来源；不再拆塔、桥、房屋为独立任务"},
    "W01": {"name": "开放平地落水", "family": "water", "template": "flat",
            "boundary": "平坦地面，无障碍、容器、坡面或缺口"},
    "W02": {"name": "独立容器侧壁释放", "family": "water", "template": "vessel",
            "boundary": "封闭底部和两侧壁，无外接通道；操作只拆侧壁"},
    "W03": {"name": "连续单坡导流", "family": "water", "template": "ramp",
            "boundary": "连续单向阶梯近似坡面，无平台、容器或额外障碍"},
    "W04": {"name": "平台阶梯落水", "family": "water", "template": "terraces",
            "boundary": "多段水平平台及垂直落差，平台长度至少三格；不是单坡"},
    "W05": {"name": "开放凹坑积水", "family": "water", "template": "pit",
            "boundary": "地下开放凹坑，无地上容器、外接通道或障碍"},
    "W06": {"name": "孤立矮挡板绕流", "family": "water", "template": "baffle",
            "boundary": "开放平地中的单个矮挡板，无侧壁、坡面、凹坑"},
    "W07": {"name": "封闭隔挡狭缝通水", "family": "water", "template": "slit",
            "boundary": "地图横向覆盖的承托隔挡，仅一个可拆狭缝；不能从侧边绕行"},
    "W08": {"name": "孤立高台边缘溢水", "family": "water", "template": "ledge",
            "boundary": "单个宽水平高台，无连级平台、容器、坡面或挡板"},
    "G01": {"name": "开放平地堆沙", "family": "sand", "template": "flat",
            "boundary": "平坦地面，无障碍、容器、坡面或缺口"},
    "G02": {"name": "独立料仓底部释放", "family": "sand", "template": "vessel",
            "boundary": "带两侧壁的独立料仓；操作只拆底部出口，无外接导向槽"},
    "G03": {"name": "开放单坡移沙", "family": "sand", "template": "ramp",
            "boundary": "连续单向阶梯近似坡面，无平台、料仓或额外障碍"},
    "G04": {"name": "平台阶梯落沙", "family": "sand", "template": "terraces",
            "boundary": "多段水平平台及垂直落差，平台长度至少三格；不是单坡"},
    "G05": {"name": "开放凹坑填沙", "family": "sand", "template": "pit",
            "boundary": "地下开放凹坑，无地上料仓、外接通道或障碍"},
    "G06": {"name": "孤立矮挡板分沙", "family": "sand", "template": "baffle",
            "boundary": "开放平地中的单个矮挡板，无侧壁、坡面、凹坑"},
    "G07": {"name": "封闭隔挡狭缝落沙", "family": "sand", "template": "slit",
            "boundary": "地图横向覆盖的承托隔挡，仅一个可拆狭缝；无料仓侧壁"},
    "G08": {"name": "孤立高台边缘落沙", "family": "sand", "template": "ledge",
            "boundary": "单个宽水平高台，无连级平台、料仓、坡面或挡板"},
}


def select(value="all"):
    result = list(TASKS) if value == "all" else value.split(",") if isinstance(value, str) else list(value)
    if not result or len(set(result)) != len(result) or any(t not in TASKS for t in result):
        raise ValueError("tasks must be distinct registered IDs")
    return result


def validate_spec(task, spec):
    if task not in TASKS or task == "T01":
        raise ValueError("synthetic specs require a registered non-legacy task")
    expected = TASKS[task]
    if spec.get("task") != task or spec.get("family") != expected["family"] or spec.get("template") != expected["template"]:
        raise ValueError("task/template assignment differs from registry")
    integer(spec["width"], "width", 3)
    integer(spec["x"], "x", 1)
    integer(spec["height"], "height", 1)
    return spec

export const phases: Record<string, string> = { precheck_target: '检查目标是否可见', precheck_candidates: '观察竞争物体与地标', precheck_decision: '寻找区分线索', decide: '选择下一步操作', execute: '执行工具或准备生成', observe_candidates: '比较场景物体外观', generate: '生成目标描述', appearance: '核对图像中的外观属性',
  recheck_feedback_landmarks: '在原视角复核反馈指定的地标属性', relook_candidates: '补看阻塞对象与其它视角', recheck_source_attributes: '复核目标属性冲突与竞争物体', ground: '按文本描述查找目标', ground_permuted: '换一种输入顺序再次检查', scene_ground: '完整场景盲定位复核', scene_ground_permuted: '交换候选编号再次盲定位', counterfactual: '改词对照检查' }
export async function readJson(response: Response) {
  const body = await response.text()
  let value
  try { value = JSON.parse(body) }
  catch {
    throw new Error(response.ok ? '服务返回的数据格式不正确，请刷新或检查后端服务。' : '请求失败（HTTP ' + response.status + '），请检查后端服务。')
  }
  if (!response.ok) throw Object.assign(new Error(typeof value.detail === 'string' ? value.detail : '请求失败（HTTP ' + response.status + '）'), { status: response.status })
  return value
}

// Bound the entire read, including the body. An Effect's cancellation must not
// be confused with a timeout: timeouts are reported and the next poll retries.
export async function readPollingJson(url: string, signal: AbortSignal, timeoutMs = 15000) {
  const controller = new AbortController()
  const cancel = () => controller.abort()
  signal.addEventListener('abort', cancel, { once: true })
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; controller.abort() }, timeoutMs)
  try {
    if (signal.aborted) controller.abort()
    return await readJson(await fetch(url, { signal: controller.signal, cache: 'no-store' }))
  } catch (e) {
    if (timedOut && !signal.aborted) throw new Error('读取超时，正在自动重连。')
    throw e
  } finally {
    clearTimeout(timer)
    signal.removeEventListener('abort', cancel)
  }
}

export function errorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)
  if (message === 'no_unique_native_camera_evidence') return '目标缺少满足当前可见性要求的图像证据或点云观测，已阻止标注；请使用新目录创建任务。'
  return /^(Failed to fetch|NetworkError.*|Load failed)$/.test(message)
    ? '服务连接中断。请确认本地服务已启动；读取恢复后此提示会消失。'
    : message
}

export function categoryLabel(category: string) {
  return ({ vehicle: '车辆', pedestrian: '行人', cyclist: '骑行者', sign: '标志牌' } as Record<string, string>)[category] ?? category
}

export function providerDiagnosisLabel(code?: string) {
  return ({ connection_not_established: '模型服务连接未建立，已自动重试；网络恢复后可重试此目标。', remote_outcome_unknown: '模型连接中断，远端调用结果尚未确认，已保留现场。', local_network_access_denied: '当前运行环境拒绝连接模型服务，请检查网络访问权限。', output_truncated: '模型输出达到长度上限，响应被截断；本条未完成检查，不作为有效标注。',
    json_mode_instruction_missing: '请求缺少 JSON 输出指令，模型服务在生成前拒绝了请求。修复配置后可在原任务中继续标注。',
    image_input_invalid: '图像输入尺寸或格式不符合模型服务要求。',
    vision_model_text_only: '本次使用的模型仅支持文本，无法核验图像；不能据此判断目标不可标注。请使用支持图像输入的模型重新标注。',
    thinking_mode_unsupported: '当前模型不支持请求指定的思考模式。',
    response_format_unsupported: '供应商不支持指定的响应格式参数；可在额度内省略此可选参数后重新请求，输出仍需严格核验。',
    request_rejected: '供应商拒绝了本条请求（HTTP 400/413/422）。原始请求和累计调用已保留；单条拒绝会跳过补选，连续三条拒绝则暂停排查。',
    image_count_rejected: '供应商拒绝了本次图像数量；保留请求并跳过该目标，不删除竞争物体或放宽核验。',
    input_length_rejected: '供应商拒绝了本次输入长度；保留请求并跳过该目标。',
    provider_account_rejected: '供应商拒绝账户或额度访问；已暂停，检查供应商账户后可在原任务继续。',
    provider_model_rejected: '供应商拒绝当前模型或访问权限；检查模型配置后可在原任务继续。' } as Record<string, string>)[code ?? ''] ?? ''
}

export function reasonLabel(reason: string) {
  if (reason === 'repeated_request_rejections') return '连续三个目标的请求被供应商拒绝，已暂停以避免继续消耗额度。请求、已有结果和累计调用均已保留，可排查后在原任务继续。'
  if (reason === 'target_excluded_after_task_deletion') return '该物体在删除旧任务时已选择不再加入候选池，已跳过。'
  return ({ unconfirmed_image_reference_view: '描述引用的来源相机或图像关系尚未获得有效证据，不能确认所指目标；该描述未通过。',
    invalid_predicate_reference: '模型声明了证据词表中不存在的定位条件，需要修正结构化声明。',
    text_claims_disagree_with_declared_predicates: '实际文本与模型声明的定位条件不一致，需要同步改写文字和条件。',
    unsupported_text_claim: '文本包含尚无已核验证据支持的说法。',
    declared_image_order_not_in_text: '模型仍声明了画面排序条件，但改写后的文本已没有对应排序描述。',
    manual_feedback_landmark_missing: '尚无完整证据支持反馈指定的地标与关系，不能用其它地标代替完成要求。',
    manual_feedback_rejected_relation_repeated: '仍使用了人工明确否定的地标关系，需要重新选择并核验证据。',
    camera_qualifier_without_image_relation: '草稿的相机限定语未通过来源检查：需有对应图像条件，或具备完整场景核验支持的真实来源外观描述。',
    unselected_image_position_in_text: '草稿新增了未选定的画面位置或街道左右线索，需要只表达已选证据。',
    unselected_visual_detail_in_text: '草稿新增了所选证据未包含的外观细节，需要重新按已选事实写作。',
    mechanical_clothing_wording: '请用 top、pants 等自然衣着措辞，不要直接输出 upper/lower clothing 这类字段名称。',
    telegraphic_camera_label: '请将相机视角写入自然英文短语，不要使用 Front view: 这样的标签式开头。',
    technical_ego_axis_comparison: '请使用清楚的空间描述，不要使用 relative to the ego 或 farther behind than 等坐标轴比较措辞；纵向排序不等于距离远近。',
    missing_reference_article: '描述省略了必要冠词；请使用 in the front view、the ego vehicle 等规范措辞。',
    telegraphic_appearance_wording: '外观描述被压缩成了不完整词组；请保留必要冠词与连接词。',
    manual_feedback_concision_missing: '描述尚未达到本轮精简目标；请减少冗余措辞并保留必要定位线索。',
    declared_joint_relation_not_in_text: '文本遗漏或改变了已选定的物体相对关系或地标限定词，需要保留完整联合描述。',
    unselected_spatial_relation_in_text: '草稿添加或反转了未被所选证据支持的空间关系，已在盲核验前拦截；人工反馈需由真实证据支持。',
    description_too_long: '草稿过长：新描述最多24个英文词，应重新选择简洁且足以唯一定位的线索。',
    technical_spatial_comparison: '草稿使用了难懂的坐标解释或前后排序比较；应改用有三维依据的简单方位或外观描述。',
    landmark_not_suitable: '参照车辆过远或缺少可靠的邻近关系证据，不能用于新描述；应改选近处地标或使用外观、自车方位。',
    target_category_omitted: '描述遗漏已确认的目标类别，不能将标志牌等目标泛称为物体。',
    incomplete_ego_direction: '目标具有明显的斜向偏移，描述遗漏了必要的前后或左右方位。',
    annotation_meta_wording: '描述含“它是目标”等标注术语，请直接使用自然的物体描述。',
    manual_feedback_target_feature_missing: '尚未落实反馈中的目标手持物特征；需重新核实，不能用包等其它附件替代。',
    manual_feedback_appearance_missing: '未能用已确认的图像依据落实反馈要求的颜色或可见背面；本轮未通过，原始观察和描述保留。',
    declared_inscription_not_in_text: '描述漏掉了所选方案中已确认的牌面文字，需要补写后重新核验。',
    camera_led_feedback_wording: '请先描述物体，再自然补充来源视角，避免用相机视角开头。',
    mechanical_surface_wording: '请自然描述标志牌的背面及其已确认颜色，不要机械复述“可见后表面”。',
    declared_ego_direction_not_in_text: '草稿遗漏了已选定的自车方位线索；来源视角不能替代“在自车前方”等真实空间关系。',
    unselected_ego_direction_in_text: '草稿添加了所选证据方案之外的自车方位，需要按已核实的方向改写。',
    joint_reference_pair_missing: '完整场景盲定位未提供支持该描述的目标与车辆地标组合。',
    joint_reference_pair_unverified: '盲定位返回的目标与车辆地标组合未通过原生身份及朝向几何核验。',
    joint_reference_landmark_identity_mismatch: '盲定位选中的地标编号与描述的地标属性不符，可能混淆了重叠物体；这不等于已证明文本有歧义。',
    joint_reference_relation_key_mismatch: '盲定位返回的关系类型与文本声明不符，需要重新核对关系方向及参照物。',
    image_visibility_scope_not_explicit: '声明的排序仅针对可见物体，但文本没有明确“可见”这一比较范围。',
    image_visibility_not_explicit_in_text: '声明使用了可见性条件，但文本没有明确说物体可见。',
    image_camera_not_explicit_in_text: '文本使用图像线索却没有明确对应的相机视角。',
    parsed_expression_not_symbolically_unique: '按实际文本解析出的条件，仍有原生 3D 竞争对象未排除。',
    image_expression_not_unique: '仍有图像竞争对象未排除，尚不能确认文本唯一指向目标。',
    unconfirmed_image_reference_evidence: '图像关系与当前相机、身份或观察证据不一致，需要重新核对。',
    unconfirmed_vehicle_heading_relation: '车辆地标的身份、朝向或唯一性证据未通过复核。',
    visual_claims_disagree_with_text: '文本里的外观属性与模型声明不一致，需要同步修正。',
    image_order_scope_not_explicit: '排序条件中的属性范围与单独解析出的目标属性不一致。',
    failed_draft_text_repeated: '重复了本轮已经核验失败的草稿，已停止重复核验，需改写区分线索。',
    scene_grounding_unsupported_claims: '完整场景盲定位未能确认文本中的部分说法，尚不支持通过。',
    scene_grounding_ambiguous: '完整场景盲定位仍发现未排除的竞争对象。',
    scene_grounding_not_unique_target: '完整场景盲定位未能唯一选中目标物体。',
    scene_grounding_invalid_candidate_ids: '完整场景盲定位返回了无效或重复的候选编号。',
    scene_grounding_identity_unverified: '完整场景盲定位所需的图文身份或原图证据尚未通过核验。',
    declared_fact_false_or_unknown: '模型声明的条件对目标不成立或证据仍不确定。',
    precheck_target_not_visible: '来源图像中无法清楚确认目标，未进入文本标注。',
    precheck_visible_but_ambiguous: '部分已尝试视角中目标已确认可见，但现有线索未能完成消歧；其它视角也未提供可用证据。',
    precheck_target_occluded_or_uncertain: '可见性预检查未通过：模型认为所选目标存在遮挡、裁切或不确定性，未生成描述。',
    precheck_target_identity_uncertain: '无法从实际可见部分确认所选物体，未生成描述；局部遮挡本身不会直接否决标注。',
    precheck_target_identity_unconfirmed: '局部可见目标尚未得到同一来源相机的身份复核，已保留记录并寻找其它证据。',
    precheck_target_view_unsuitable: '当前来源视角未满足该条记录的身份与图像证据要求，未生成描述。',
    precheck_source_visibility_conflict: '同一来源相机的可见性观察冲突，已停止标注，不作为训练标签。',
    no_suitable_sign_face: '没有可确认朝向相机的信息面的标志牌视图，未选为本轮训练目标。',
    precheck_target_not_distinct: '当前系统尚未找到能排除其它物体的描述线索，因此未生成文本；不代表该物体无法标注。',
    precheck_no_distinguishing_cues: '本次已核验线索尚不足以完成唯一定位，已保留目标及失败证据；可补充视角或描述线索继续核验。',
    precheck_native_identity_ambiguous: '目标与另一个原生 3D 框高度重叠，现有衣着、外观及空间关系仍无法区分两个身份；已保留原始框和冲突证据，不能将同一可见物体强行标给其中一个框。',
    precheck_no_reliable_visual_feature: '图像未提供可靠的类型或外观特征，未进入文本标注。',
    precheck_invalid_scene_observation: '场景观察返回了无效或重复的物体索引，未进入文本标注。',
    sample_identity_binding_failed: '物体、实际采集帧或换帧凭据不一致，已保留记录并停止将此样本计为成功，其余目标可继续处理。',
    two_round_blind_scene_grounding_passed: '两轮定位检查均通过，描述已保存并计入任务完成数量。',
    all_oracle_checks_passed: '自动定位检查已通过，描述已保存。独立质量评估另行进行。',
    agent_abstained: '系统判断现有证据或预算不足，结束此次尝试。',
    target_spatially_ambiguous: '现有空间线索仍无法可靠区分目标，已跳过，不作为训练标签。',
    description_validation_failed: '描述中的属性或区分线索未通过核验，未取得通过检查的修复版本，不作为训练标签。',
    manual_feedback_distance_missing: '草稿没有落实反馈要求的近距离条件；不能用左右或前后方位替代。',
    ego_distance_claim_mismatch: '距离措辞与所选几何条件不一致。',
    ego_distance_not_supported: '原生几何未确认该近距离条件，不能凭图像印象使用。',
    manual_feedback_distance_unsupported: '反馈要求描述与自车的距离，但当前尚未定义近／远的几何核验标准；不能用左右方位替代。',
    manual_feedback_retained_phrase_missing: '草稿未保留反馈明确指定的原文。',
    manual_feedback_motion_unsupported: '反馈要求描述运动状态或行驶方向，但当前单帧标注尚无可用的运动核验链；本轮未完成，不会省略要求后报成功。',
    manual_feedback_spatial_missing: '反馈要求的空间线索尚未落实到有证据支持的描述中；来源视角不能代替位置或关系。',
    insufficient_validation_budget: '剩余调用额度不足以完成生成和核验，已停止，避免留下未检查的标注。',
    model_call_budget_exhausted: '本次模型调用额度已用完，未继续发送请求。',
    no_useful_action_within_budget: '现有线索无法区分目标，且没有可执行的补充或修复操作。此次标注已停止，样本和原因已保存。',
    'model_call:invalid_output': '模型响应不符合要求，系统已记录错误并停止此次执行。',
    'model_call:api_failed': '模型服务请求失败。限流或暂时性服务错误已按额度退避重试，仍失败的目标保留原因；认证或配置错误会暂停任务。',
    'model_call:interrupted_response': '进程中断时调用结果未知，已保留原记录并跳过此目标，避免重复提交；其余目标继续处理。',
    restart_configuration_or_input_changed: '中断恢复时发现代码、配置或输入发生变化，已暂停；点击“继续标注”保留完整样本并在原任务中恢复。',
    interrupted_response_outcome_unknown: '中断时模型请求结果未知，已保留调用记录并跳过该物体，避免重复发送。',
    response_deadline_exceeded: '模型响应超过时长限制，已保留调用记录并跳过此目标，其余目标继续处理。',
    'model_call:connection_failed': '模型服务连接失败，已保留进度；确认未发送的请求可在原预算内重试。',
    provider_temporarily_unavailable: '连续两个目标在自动重试后仍无法连接模型服务，已停止后续请求。网络恢复后可重试失败目标并继续。',
    oracle_passed_counterfactual_not_applicable: '自动定位检查已通过，描述已保存并计入任务完成数量。本条无需改词对照检查；独立质量评估另行进行。',
    oracle_passed_counterfactual_budget_incomplete: '自动定位检查已通过，描述已保存。剩余模型调用次数不足，未完成改词对照检查；独立质量评估另行进行。',
    oracle_and_counterfactual_checks_passed: '自动定位和改词对照检查均通过，描述已保存。独立质量评估另行进行。',
    oracle_passed_counterfactual_reader_inconsistent: '自动定位检查已通过，描述已保存。额外的改词对照检查结果不一致，相关记录已保留，供后续质量评估。',
  } as Record<string, string>)[reason] ?? `处理说明：${reason}。详细信息见执行记录。`
}

export function sourcePath(file: string | null | undefined) {
  if (!file) return '来源文件未记录'
  const filename = file.split(/[\\/]/).pop() ?? ''
  const root = filename.startsWith('validation_') ? 'waymo_valset' : 'waymo_trainset'
  return `${root}/${file}`
}

export const objectName = (id: string) => `物体 ${Number(id.slice(1))}`
export function recordStatus(item: {status:string; run_status?:string; manual_approved?:boolean; manual_review?:{verdict:string}|null; provider_diagnosis?:string; reason?:string|null}, task?:{status:string; control?:string|null}) {
  if (item.manual_review?.verdict === 'discarded') return {label:'已废弃', tone:'neutral'}
  if (item.manual_review?.verdict === 'rejected') return {label:'人工审核未通过', tone:'failed'}
  if (item.manual_approved) return {label:'人工审批通过', tone:'passed'}
  const status=item.run_status ?? item.status
  if (['calibration_hold','auto_pass'].includes(status)) return {label:'自动核验通过', tone:'passed'}
  const active=['pending','ready','preparing','prepared','running','paused','stopped'].includes(status)
  if (active && task?.control === 'pause') return {label:'正在暂停', tone:'waiting'}
  if (active && task?.control === 'stop') return {label:'正在停止', tone:'waiting'}
  if (active && (task?.status === 'paused' || status === 'paused')) return {label:'已暂停', tone:'waiting'}
  if (active && (task?.status === 'stopped' || status === 'stopped')) return {label:'已停止', tone:'neutral'}
  if (item.provider_diagnosis === 'remote_outcome_unknown' || item.reason === 'interrupted_response_outcome_unknown' || item.reason === 'model_call:interrupted_response') return {label:'连接中断 · 结果待确认', tone:'error'}
  if (['connection_not_established','local_network_access_denied'].includes(item.provider_diagnosis ?? '') || item.reason === 'model_call:connection_failed') return {label:'网络连接失败', tone:'error'}
  if (item.reason === 'response_deadline_exceeded') return {label:'模型响应超时', tone:'error'}
  if (item.reason?.startsWith('model_call:') || item.provider_diagnosis) return {label:'模型请求异常', tone:'error'}
  if (status === 'failed' || status === 'blocked') return {label:'执行出错', tone:'error'}
  if (status === 'needs_evidence') return {label:'未通过核验', tone:'failed'}
  if (status === 'abstained') return {label:'未完成标注', tone:'failed'}
  if (status === 'excluded') return {label:'预检查未通过', tone:'failed'}
  if (status === 'budget_skipped') return {label:'调用次数不足 · 未执行', tone:'waiting'}
  if (status === 'not_needed') return {label:'未执行', tone:'neutral'}
  if (status === 'preparing') return {label:'正在准备数据', tone:'running'}
  if (status === 'running') return {label:'执行中', tone:'running'}
  if (status === 'prepared') return {label:'数据已准备', tone:'waiting'}
  if (status === 'pending' || status === 'ready') return {label:'等待处理', tone:'waiting'}
  return {label:'状态待确认', tone:'neutral'}
}
export function isAnnotationResult(item: { status: string; run_status?: string; expression?: string | null; manual_approved?:boolean; manual_review?: { verdict: string } | null }): boolean {
  return !['rejected', 'discarded'].includes(item.manual_review?.verdict ?? '')
    && !!item.expression?.trim() && (item.manual_approved===true||['calibration_hold', 'auto_pass'].includes(item.run_status ?? item.status))
}

export function isProcessingRecord(item: { status: string; packet_id?: string | null; run_id?: string | null }): boolean {
  return !!(item.packet_id || item.run_id) || ['preparing', 'running', 'prepared', 'calibration_hold',
    'auto_pass', 'abstained', 'needs_evidence', 'failed', 'excluded', 'reused'].includes(item.status)
}

export const manualReviewLabels = { accepted: '审核通过', rejected: '审核不通过', discarded: '已废弃' }

export const automaticCheckLabels = {
  passed: '自动核验通过', failed: '自动核验未通过', pending: '自动核验未完成', unknown: '核验状态未读取',
}
export function automaticCheckState(sample: { automatic_passed?: boolean; status: string; can_review: boolean }): keyof typeof automaticCheckLabels {
  if (sample.automatic_passed === true) return 'passed'
  if (sample.automatic_passed !== false) return 'unknown'
  return !sample.can_review && ['running', 'paused', 'stopped', 'pending', 'preparing', 'prepared'].includes(sample.status)
    ? 'pending' : 'failed'
}

export const sampleFilterLabels = {
  all: '全部描述', usable: '当前可用', passed: '自动核验通过', manual_approved: '人工审批通过',
  failed: '自动核验未通过', discarded: '已废弃', pending: '自动核验未完成', unknown: '核验状态未读取',
}
export type SampleFilter = keyof typeof sampleFilterLabels
export function matchesSampleFilter(sample: { automatic_passed?: boolean; status: string; can_review: boolean;
  manual_approved?: boolean; usable?: boolean; manual_review?: { verdict: string } | null }, filter: SampleFilter): boolean {
  if (filter === 'all') return true
  const discarded = sample.manual_review?.verdict === 'discarded' || sample.status === 'discarded'
  // Human dispositions take precedence in the working lists. The original
  // automatic verdict remains available through automaticCheckState/details.
  if (discarded) return filter === 'discarded'
  if (filter === 'usable') return sample.usable === true
  if (sample.manual_approved === true) return filter === 'manual_approved'
  return automaticCheckState(sample) === filter
}

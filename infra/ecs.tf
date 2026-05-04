resource "aws_cloudwatch_log_group" "ecs" {
  name              = local.log_group
  retention_in_days = 7
}

resource "aws_ecs_cluster" "main" {
  name = local.cluster_name

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

# ----- ALB -----
resource "aws_lb" "main" {
  name               = local.alb_name
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
  idle_timeout       = 60
}

resource "aws_lb_target_group" "backend" {
  name        = local.tg_backend
  port        = 5001
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/api/health"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_target_group" "frontend" {
  name        = local.tg_frontend
  port        = 80
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/"
    matcher             = "200-399"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Default → frontend (anything that isn't /api/*)
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}

# /api/* → backend
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# ----- Backend task + service -----
resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name_prefix}-backend-${local.unique_suffix}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = "arn:aws:iam::${local.account_id}:role/LabRole"
  task_role_arn            = "arn:aws:iam::${local.account_id}:role/LabRole"

  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = "${aws_ecr_repository.backend.repository_url}:${var.container_image_tag}"
      essential = true

      portMappings = [{
        containerPort = 5001
        hostPort      = 5001
        protocol      = "tcp"
      }]

      environment = [
        { name = "NODE_ENV", value = "production" },
        { name = "PORT", value = "5001" }
      ]

      secrets = [
        { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn },
        { name = "JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt_secret.arn }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = local.region
          awslogs-stream-prefix = "backend"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "wget -qO- http://localhost:5001/api/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])
}

resource "aws_ecs_service" "backend" {
  name             = "${local.service_prefix}-backend"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.backend.arn
  desired_count    = 2
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 5001
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]
}

# ----- Client (nginx) task + service -----
resource "aws_ecs_task_definition" "frontend" {
  family                   = "${local.name_prefix}-frontend-${local.unique_suffix}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = "arn:aws:iam::${local.account_id}:role/LabRole"
  task_role_arn            = "arn:aws:iam::${local.account_id}:role/LabRole"

  container_definitions = jsonencode([
    {
      name      = "frontend"
      image     = "${aws_ecr_repository.frontend.repository_url}:${var.container_image_tag}"
      essential = true

      portMappings = [{
        containerPort = 80
        hostPort      = 80
        protocol      = "tcp"
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = local.region
          awslogs-stream-prefix = "frontend"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "wget -qO- http://localhost/ || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 15
      }
    }
  ])
}

resource "aws_ecs_service" "frontend" {
  name             = "${local.service_prefix}-frontend"
  cluster          = aws_ecs_cluster.main.id
  task_definition  = aws_ecs_task_definition.frontend.arn
  desired_count    = 2
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.frontend_ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "frontend"
    container_port   = 80
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]
}
